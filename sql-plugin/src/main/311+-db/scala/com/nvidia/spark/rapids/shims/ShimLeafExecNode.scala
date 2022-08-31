/*
 * Copyright (c) 2022, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.nvidia.spark.rapids.shims

import org.apache.spark.sql.catalyst.plans.logical.Statistics
import org.apache.spark.sql.execution.LeafExecNode
import org.apache.spark.sql.execution.datasources.v2.DataSourceV2ScanExecBase

trait ShimLeafExecNode extends LeafExecNode {
  override def computeStats(): Statistics = {
    Statistics(
      sizeInBytes = Long.MaxValue
    )
  }
}

trait ShimDataSourceV2ScanExecBase extends DataSourceV2ScanExecBase {
  override def computeStats(): Statistics = {
    Statistics(
      sizeInBytes = Long.MaxValue
    )
  }
}