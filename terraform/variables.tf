variable "aws_region" {
  type    = string
  default = "eu-west-2"
}

variable "allowed_cidr_blocks" {
  description = "CIDR blocks allowed to reach RDS directly (your own IP as x.x.x.x/32). Never leave this as 0.0.0.0/0."
  type        = list(string)
}

variable "github_repository" {
  description = "owner/repo allowed to assume the CI role via OIDC."
  type        = string
  default     = "daria-dot/ticket-triage"
}

# Both from: gh api repos/OWNER/REPO --jq '{repo: .id, owner: .owner.id}'
variable "github_owner_id" {
  description = "Immutable numeric GitHub owner ID, as it appears in the OIDC subject claim."
  type        = string
  default     = "186747603"
}

variable "github_repository_id" {
  description = "Immutable numeric GitHub repository ID, as it appears in the OIDC subject claim."
  type        = string
  default     = "1365824571"
}

variable "api_image_tag" {
  description = "ECR image tag to serve from SageMaker. Empty means no endpoint is created at all."
  type        = string
  default     = ""
}

variable "model_artifact_key" {
  description = "Key of model.tar.gz within the artifacts bucket."
  type        = string
  default     = "models/baseline/model.tar.gz"
}

variable "create_rds" {
  description = "Whether the billable RDS instance should exist. Off by default so an unrelated apply can't quietly provision a database."
  type        = bool
  default     = false
}

variable "postgres_db" {
  type    = string
  default = "triage"
}

variable "postgres_user" {
  type    = string
  default = "triage"
}
