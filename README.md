# Deploying Detectron2 Model on CVAT on AWS

## Prerequisites

- **CVAT** : [How to install CVAT](https://docs.cvat.ai/docs/administration/basics/installation/)
- **Automatic annotation tool** : [How to install the automatic annotation tool](https://docs.cvat.ai/docs/administration/advanced/installation_automatic_annotation/)
- **GIT** : Git must be installed on the system.

## Deploy the model

- First, you need to clone this [repository](https://github.com/JalilBNH/mask-rcnn-cvat.git) into your AWS instance
- You can deploy the model with this command :
```console
ubuntu@ip:~$ cvat/serverless/deploy_cpu.sh mask-rcnn-cvat/mask_rcnn/
```
*If you want to deploy a model that produce less point on the segmentation, you can deploy 'mask_rcnn_sampl' and you can modify the number of point you want in the main.py*