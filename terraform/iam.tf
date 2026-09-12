# GitHub Actions authenticates via OIDC and assumes this role for the duration
# of a job, so no long-lived AWS keys are stored as repository secrets.

locals {
  github_owner = split("/", var.github_repository)[0]
  github_repo  = split("/", var.github_repository)[1]

  github_repository_with_ids = join("/", [
    "${local.github_owner}@${var.github_owner_id}",
    "${local.github_repo}@${var.github_repository_id}",
  ])
}

resource "aws_iam_openid_connect_provider" "github" {
  url             = "https://token.actions.githubusercontent.com"
  client_id_list  = ["sts.amazonaws.com"]
  thumbprint_list = ["6938fd4d98bab03faadb97b34396831e3780aea1"]
}

resource "aws_iam_role" "github_actions" {
  name = "ticket-triage-github-actions"

  # Scoped to pushes to main on this one repo: PR builds don't push images,
  # so they have no reason to be able to assume this.
  #
  # GitHub embeds immutable numeric owner/repo IDs in the subject claim
  # (repo:owner@<owner_id>/name@<repo_id>:ref:...) so that deleting a repo and
  # recreating it under the same name cannot inherit the original's trust.
  # Whether it does so is a GitHub-side setting, so both forms are listed --
  # a list in StringEquals matches any entry, keeping this an exact match
  # rather than a wildcard.
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect    = "Allow"
        Action    = "sts:AssumeRoleWithWebIdentity"
        Principal = { Federated = aws_iam_openid_connect_provider.github.arn }
        Condition = {
          StringEquals = {
            "token.actions.githubusercontent.com:aud" = "sts.amazonaws.com"
            "token.actions.githubusercontent.com:sub" = [
              "repo:${var.github_repository}:ref:refs/heads/main",
              "repo:${local.github_repository_with_ids}:ref:refs/heads/main",
            ]
          }
        }
      },
    ]
  })
}

resource "aws_iam_role_policy" "github_actions_ecr_push" {
  name = "ecr-push"
  role = aws_iam_role.github_actions.id

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
          "ecr:InitiateLayerUpload",
          "ecr:UploadLayerPart",
          "ecr:CompleteLayerUpload",
          "ecr:PutImage",
        ]
        Resource = aws_ecr_repository.api.arn
      },
    ]
  })
}
