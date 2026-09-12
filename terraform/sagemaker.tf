# Serverless inference rather than a real-time endpoint: it scales to zero, so
# an endpoint nobody is calling costs nothing, where a real-time variant bills
# per second around the clock. The trade is a few seconds of cold start, which
# is irrelevant for a portfolio endpoint and would matter for real traffic.
#
# Everything here is gated on api_image_tag being set. SageMaker models are
# immutable, so the tag is part of the model name and a new image produces a
# new model rather than mutating one in place.

locals {
  sagemaker_enabled = var.api_image_tag != ""
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

    serverless_config {
      # Headroom for loading scikit-learn, MLflow and a 20k-feature vectoriser.
      # Serverless bills per GB-second while a request is in flight, so this
      # costs more per call but nothing at all while idle.
      memory_size_in_mb = 3072
      max_concurrency   = 1
    }
  }
}

resource "aws_sagemaker_endpoint" "api" {
  count = local.sagemaker_enabled ? 1 : 0

  name                 = "ticket-triage"
  endpoint_config_name = aws_sagemaker_endpoint_configuration.api[0].name
}
