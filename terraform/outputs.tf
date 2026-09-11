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
