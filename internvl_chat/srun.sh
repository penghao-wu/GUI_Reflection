PARTITION=cc4cdf6c-944a-4c46-a9de-3d508a06c4dd #AMP
PARTITION=amplarge #amplarge
# PARTITION=b7c081ea-ab5a-4278-ab4a-c51bc222de13 #H100_SHARE
# # PARTITION=r1-m1
# PARTITION=r1-m1-large
# PARTITION=err-nodes

WORKSPACE=a58d023b-de76-475f-89c2-7e50f7aa3c7a
CONTAINTER=registry.ms-sc-01.maoshanwangtech.com/studio-aicl/ubuntu20.04-py3.10-cuda11.8-cudnn8-transformer4.28.0:master-20230626-172512-32302
MOUNT=ce3b1174-f6eb-11ee-a372-82d352e10aed:/mnt/afs,1f29056c-c3f2-11ee-967e-2aea81fd34ba:/mnt/afs2,047443d2-c3f2-11ee-a5f9-9e29792dec2f:/mnt/afs1
SPEC=N6lS.Iu.I80.8
# SPEC=N6lS.Iq.I10.8

SCRIPT=$(pwd)/train_scripts/offline_sft.sh

sco acp jobs create \
--workspace-name  $WORKSPACE -p $PARTITION \
--container-image-url $CONTAINTER \
--storage-mount $MOUNT \
--worker-spec $SPEC \
--training-framework pytorch \
--worker-nodes 4 \
-j test \
--command="bash $SCRIPT"