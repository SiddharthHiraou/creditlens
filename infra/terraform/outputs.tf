output "api_url" {
  description = "Public URL of the scoring API."
  value       = "http://${aws_lb.main.dns_name}"
}

output "ecr_repository_url" {
  description = "Push target for the API image."
  value       = aws_ecr_repository.api.repository_url
}

output "artifacts_bucket" {
  description = "S3 bucket holding model artifacts and monitoring exports."
  value       = aws_s3_bucket.artifacts.id
}

output "database_endpoint" {
  description = "RDS endpoint. Not publicly reachable."
  value       = aws_db_instance.audit.endpoint
  sensitive   = true
}
