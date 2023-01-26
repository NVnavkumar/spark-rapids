/*
 * Copyright (c) 2023, NVIDIA CORPORATION.
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

package com.nvidia.spark.rapids.delta.shims

import org.apache.spark.sql.delta.{DeltaLog, DeltaUDF, Snapshot}
import org.apache.spark.sql.expressions.UserDefinedFunction

class Delta22PlusRuntimeShim extends DeltaRuntimeShim {
  override def stringFromStringUdf(f: String => String): UserDefinedFunction = {
    DeltaUDF.stringFromString(f)
  }

  override def unsafeVolatileSnapshotFromLog(deltaLog: DeltaLog): Snapshot = {
    deltaLog.unsafeVolatileSnapshot
  }
}