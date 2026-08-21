# Terraform — CreditLens

Provisions the smallest set of AWS resources that runs the scoring API with a
real database behind a load balancer: **ECR, ECS Fargate, RDS Postgres, ALB, S3**.

```bash
cd infra/terraform
terraform init
terraform plan  -var="db_password=$DB_PASSWORD" -var="api_key=$API_KEY"
terraform apply -var="db_password=$DB_PASSWORD" -var="api_key=$API_KEY"

# then push the image and force a new deployment
aws ecr get-login-password | docker login --username AWS --password-stdin "$(terraform output -raw ecr_repository_url)"
docker build -t "$(terraform output -raw ecr_repository_url):v1" -f ../docker/Dockerfile.api ../..
docker push "$(terraform output -raw ecr_repository_url):v1"
terraform apply -var="image_tag=v1" -var="db_password=$DB_PASSWORD" -var="api_key=$API_KEY"

curl -s "$(terraform output -raw api_url)/health"
```

## Choices worth defending

**`/ready` is the ALB health check, not `/health`.** An instance whose database
is unreachable is *up* but cannot serve a decision it can audit. It should leave
rotation.

**A 90-second health check grace period.** The model and the TreeSHAP explainer
take roughly 20 seconds to load. Without the grace period ECS kills tasks that
are simply still starting, and the service never converges.

**ECR tags are immutable.** A tag must always mean the same image, or a rollback
is not a rollback.

**Separate execution and task roles.** The execution role pulls images and writes
logs; the task role is the only thing the application itself can do, and it is
scoped to reading one S3 bucket.

**No NAT gateway.** Tasks run in public subnets with public IPs, reachable only
from the ALB security group. A NAT gateway is roughly $32/month for a demo that
does not need it. The database stays in private subnets regardless.

## NOT INCLUDED — deliberately

This is a minimal module, not a production reference. Missing, and named rather
than half-built:

- **Secrets Manager.** `db_password` and `api_key` are passed as sensitive
  variables and land in the task definition as plaintext environment variables.
  Real deployments inject them by secret ARN.
- **TLS.** HTTP only. HTTPS needs an ACM certificate and a domain.
- **Multi-AZ RDS.** Single instance; durability through backups.
- **Autoscaling.** Fixed `desired_count`.
- **WAF, CloudFront, VPC flow logs, GuardDuty.**
- **Remote state.** Local state file. Real use needs S3 with DynamoDB locking.
- **Redis (ElastiCache).** The feature cache falls back to in-process, which
  means each task warms its own copy. Fine for a demo, wasteful at scale.

A Terraform module that looks production-ready and is not is worse than one that
is honest about its scope.
