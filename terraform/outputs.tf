output "vpc_id" {
  value = aws_vpc.main.id
}

output "public_subnet_ids" {
  value = aws_subnet.public[*].id
}

output "rds_security_group_id" {
  value = aws_security_group.rds.id
}

output "ecr_repository_url" {
  value = aws_ecr_repository.api.repository_url
}

output "github_actions_role_arn" {
  value = aws_iam_role.github_actions.arn
}

output "db_endpoint" {
  description = "Null while create_rds is false and no database exists."
  value       = one(aws_db_instance.main[*].endpoint)
}

output "db_password" {
  value     = random_password.db.result
  sensitive = true
}

output "artifacts_bucket" {
  value = aws_s3_bucket.artifacts.id
}

output "sagemaker_endpoint_name" {
  description = "Null while api_image_tag is unset and no endpoint exists."
  value       = one(aws_sagemaker_endpoint.api[*].name)
}
