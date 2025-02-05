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
/*** spark-rapids-shim-json-lines
{"spark": "340"}
{"spark": "341"}
{"spark": "341db"}
{"spark": "350"}
spark-rapids-shim-json-lines ***/
package com.nvidia.spark.rapids.shims

import ai.rapids.cudf.{ColumnVector, Scalar}
import com.nvidia.spark.rapids.Arm.withResource
import com.nvidia.spark.rapids.GpuCast

import org.apache.spark.sql.catalyst.json.GpuJsonUtils

object GpuJsonToStructsShim {

  def castJsonStringToDate(input: ColumnVector, options: Map[String, String]): ColumnVector = {
    GpuJsonUtils.optionalDateFormatInRead(options) match {
      case None =>
        // legacy behavior
        withResource(Scalar.fromString(" ")) { space =>
          withResource(input.strip(space)) { trimmed =>
            GpuCast.castStringToDate(trimmed)
          }
        }
      case Some("yyyy-MM-dd") =>
        GpuCast.convertDateOrNull(input, "^[0-9]{4}-[0-9]{2}-[0-9]{2}$", "%Y-%m-%d")
      case other =>
        // should be unreachable due to GpuOverrides checks
        throw new IllegalStateException(s"Unsupported dateFormat $other")
    }
  }

  def castJsonStringToTimestamp(input: ColumnVector,
      options: Map[String, String]): ColumnVector = {
    options.get("timestampFormat") match {
      case None =>
        // legacy behavior
        withResource(Scalar.fromString(" ")) { space =>
          withResource(input.strip(space)) { trimmed =>
            // from_json doesn't respect ansi mode
            GpuCast.castStringToTimestamp(trimmed, ansiMode = false)
          }
        }
      case Some("yyyy-MM-dd'T'HH:mm:ss[.SSS][XXX]") =>
        GpuCast.convertTimestampOrNull(input,
          "^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}(\\.[0-9]{1,6})?Z?$", "%Y-%m-%d")
      case other =>
        // should be unreachable due to GpuOverrides checks
        throw new IllegalStateException(s"Unsupported timestampFormat $other")
      }
  }
}