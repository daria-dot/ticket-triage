variable "aws_region" {
  type    = string
  default = "eu-west-2"
}

variable "allowed_cidr_blocks" {
  description = "CIDR blocks allowed to reach RDS directly (your own IP as x.x.x.x/32). Never leave this as 0.0.0.0/0."
  type        = list(string)
}
