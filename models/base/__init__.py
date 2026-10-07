#!/usr/bin/env python3
# Copyright (C) Alibaba Group Holding Limited. 

import models.module_zoo
from models.base.base_blocks import BaseHead, Base3DStem
import models.base.transformer
import models.base.slowfast
import models.base.few_shot
import models.base.semantic_alignment_supervised_supervised
import models.base.semantic_alignment_few_shot
from OTAM.model import OTAM_VIT
from models.base.base_blocks import HEAD_REGISTRY

HEAD_REGISTRY.register()(OTAM_VIT)
