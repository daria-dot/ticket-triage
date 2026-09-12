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

variable "create_sagemaker" {
  description = "Whether the billable SageMaker endpoint should exist. Off by default: it bills per second for as long as it exists."
  type        = bool
  default     = false
}

variable "sagemaker_instance_type" {
  description = "Cheapest x86 instance with a non-zero quota in eu-west-2. The cheaper c6g family is Graviton and won't run an amd64 image."
  type        = string
  default     = "ml.c5.large"
}

variable "api_image_tag" {
  description = "ECR image tag to serve from SageMaker. Bump deliberately to promote a new image, the way model_version promotes a new model."
  type        = string
  default     = "bafc5c3eb7da5f492c081facfeda5fe744c55b5e"
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
