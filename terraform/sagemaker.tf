# Real-time inference. Serverless would be the better fit -- it scales to zero,
# so an idle endpoint costs nothing, where this bills per second for as long as
# it exists -- but serverless endpoints cannot be created in this account.
#
# That was established rather than assumed. Creation fails with a bare "Request
# to service failed", producing no container logs at all, and it fails
# identically with no model artifact attached. The image itself was pulled from
# ECR and run locally exactly as SageMaker runs it (`docker run <image> serve`,
# model mounted at /opt/ml/model) and served /ping and /invocations correctly;
# every IAM action simulates as allowed; the manifest is a single amd64 v2
# manifest, not a multi-arch list; and serverless quotas are non-zero. The same
# image, artifact and role then created a real-time endpoint first time and
# returned correct predictions. The fault is on the AWS side.
#
# Because this bills continuously, it is gated behind create_sagemaker and
# defaults to off, so an apply made for an unrelated change provisions nothing.
# SageMaker models are immutable, so the image tag forms part of the model name
# and a new image produces a new model rather than mutating one in place.

locals {
  sagemaker_enabled = var.create_sagemaker && var.api_image_tag != ""
  model_name        = "ticket-triage-${substr(var.api_image_tag, 0, 12)}"
}

resource "aws_iam_role" "sagemaker_execution" {
  name = "ticket-triage-sagemaker-execution"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect    = "Allow"
        Action    = "sts:AssumeRole"
        Principal = { Service = "sagemaker.amazonaws.com" }
      },
    ]
  })
}

resource "aws_iam_role_policy" "sagemaker_execution" {
  name = "pull-image-and-model"
  role = aws_iam_role.sagemaker_execution.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect   = "Allow"
        Action   = "ecr:GetAuthorizationToken"
        Resource = "*" # This action does not support resource-level permissions.
      },
      {
        Effect = "Allow"
        Action = [
          "ecr:BatchCheckLayerAvailability",
          "ecr:BatchGetImage",
          "ecr:GetDownloadUrlForLayer",
        ]
        Resource = aws_ecr_repository.api.arn
      },
      {
        Effect   = "Allow"
        Action   = "s3:GetObject"
        Resource = "${aws_s3_bucket.artifacts.arn}/*"
      },
      {
        Effect = "Allow"
        Action = [
          "logs:CreateLogGroup",
          "logs:CreateLogStream",
          "logs:PutLogEvents",
        ]
        Resource = "arn:aws:logs:${var.aws_region}:${data.aws_caller_identity.current.account_id}:log-group:/aws/sagemaker/*"
      },
    ]
  })
}

resource "aws_sagemaker_model" "api" {
  count = local.sagemaker_enabled ? 1 : 0

  name               = local.model_name
  execution_role_arn = aws_iam_role.sagemaker_execution.arn

  primary_container {
    image          = "${aws_ecr_repository.api.repository_url}:${var.api_image_tag}"
    model_data_url = "s3://${aws_s3_bucket.artifacts.id}/${var.model_artifact_key}"
  }
}

resource "aws_sagemaker_endpoint_configuration" "api" {
  count = local.sagemaker_enabled ? 1 : 0

  name = local.model_name

  production_variants {
    variant_name = "AllTraffic"
    model_name   = aws_sagemaker_model.api[0].name

    # ml.c5.large is the cheapest x86 instance with a non-zero quota here. The
    # cheaper ml.c6g family is Graviton, and this image is amd64.
    instance_type          = var.sagemaker_instance_type
    initial_instance_count = 1
  }
}

resource "aws_sagemaker_endpoint" "api" {
  count = local.sagemaker_enabled ? 1 : 0

  name                 = "ticket-triage"
  endpoint_config_name = aws_sagemaker_endpoint_configuration.api[0].name
}
