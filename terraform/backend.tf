# Backend config can't reference variables (a Terraform constraint), so the
# bootstrap bucket/table names are hardcoded here -- see terraform/bootstrap/.
terraform {
  required_version = ">= 1.5"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
    random = {
      source  = "hashicorp/random"
      version = "~> 3.6"
    }
  }

  backend "s3" {
    bucket         = "ticket-triage-terraform-state-961341523002"
    key            = "ticket-triage/terraform.tfstate"
    region         = "eu-west-2"
    dynamodb_table = "ticket-triage-terraform-locks"
    encrypt        = true
  }
}

provider "aws" {
  region = var.aws_region
}
