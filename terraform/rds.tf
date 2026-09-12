# The deployed environment's Postgres. Local docker-compose Postgres remains
# the authoritative dev store (see Phase 3), so this instance is expected to be
# created on demand and destroyed at the end of a session -- the single largest
# avoidable cost in this project after the SageMaker endpoint.
#
# It is therefore gated behind create_rds, defaulting to off. Without that, an
# apply made for an unrelated change would silently rebuild the database and
# start billing again, which is exactly the accident this project is trying to
# avoid. `make infra-up` flips it on; `make infra-down` flips it back.

resource "random_password" "db" {
  length = 32
  # RDS rejects /, @, ", and spaces in master passwords.
  override_special = "!#$%&*()-_=+[]{}<>:?"
}

resource "aws_db_subnet_group" "main" {
  name       = "ticket-triage"
  subnet_ids = aws_subnet.public[*].id
}

resource "aws_db_instance" "main" {
  count = var.create_rds ? 1 : 0

  identifier     = "ticket-triage"
  engine         = "postgres"
  engine_version = "16"
  instance_class = "db.t4g.micro"

  allocated_storage = 20
  storage_type      = "gp3"
  storage_encrypted = true

  db_name  = var.postgres_db
  username = var.postgres_user
  password = random_password.db.result

  db_subnet_group_name   = aws_db_subnet_group.main.name
  vpc_security_group_ids = [aws_security_group.rds.id]
  publicly_accessible    = true

  # Backup storage up to the size of the instance is free; one day is enough
  # for a store that gets torn down between sessions anyway.
  backup_retention_period = 1

  # Enhanced monitoring and Performance Insights both bill separately.
  monitoring_interval          = 0
  performance_insights_enabled = false

  # Nothing here outlives a session, and deletion protection would block the
  # teardown this project depends on for keeping credit burn near zero.
  skip_final_snapshot = true
  deletion_protection = false
}
