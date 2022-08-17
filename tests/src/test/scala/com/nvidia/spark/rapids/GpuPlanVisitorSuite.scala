
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

package com.nvidia.spark.rapids

import java.io.File

import org.apache.spark.SparkConf
import org.apache.spark.internal.Logging
import org.apache.spark.sql.catalyst.plans.logical.LogicalPlan
import org.apache.spark.sql.{Dataset, Row, SaveMode, SparkSession}
import org.apache.spark.sql.execution.adaptive.{AdaptiveSparkPlanExec, AdaptiveSparkPlanHelper}
import org.apache.spark.sql.execution.SparkPlan
import org.apache.spark.sql.functions.col
import org.apache.spark.sql.internal.SQLConf

class GpuPlanVisitorSuite
  extends SparkQueryCompareTestSuite
  with AdaptiveSparkPlanHelper
  with FunSuiteWithTempDir
  with Logging {

  private def getPlansFromQuery(spark: SparkSession, query: String,
      isAdaptive: Boolean): (LogicalPlan, SparkPlan) = {
    val df = spark.sql(query)
    val optimized = df.queryExecution.optimizedPlan
    val planBefore = df.queryExecution.executedPlan
    if (!isAdaptive) {
      (optimized, planBefore)
    } else {
      df.collect()
      val planAfter = df.queryExecution.executedPlan
      val adaptivePlan = planAfter.asInstanceOf[AdaptiveSparkPlanExec].executedPlan
      (optimized, adaptivePlan)
    }
  }
  
  test("stats from testData dataframe") {
    val conf = new SparkConf()
      .set(SQLConf.ADAPTIVE_EXECUTION_ENABLED.key, "false")

    val (logicalStats, gpuStats) = withGpuSparkSession(spark => {
      testData(spark)
      val (logical, gpuPlan) = getPlansFromQuery(spark, "select * from testData", false)
      (logical.computeStats(), gpuPlan.computeStats())
    })

    assert(logicalStats.sizeInBytes == gpuStats.sizeInBytes)
  }

  /** Ported from org.apache.spark.sql.test.SQLTestData */
  private def testData(spark: SparkSession) {
    import spark.implicits._
    val data: Seq[(Int, String)] = (1 to 100).map(i => (i, i.toString))
    val df = data.toDF("key", "value")
        .repartition(col("key"))
    registerAsParquetTable(spark, df, "testData")  }

  private def registerAsParquetTable(spark: SparkSession, df: Dataset[Row], name: String) {
    val path = new File(TEST_FILES_ROOT, s"$name.parquet").getAbsolutePath
    df.write
        .mode(SaveMode.Overwrite)
        .parquet(path)
    spark.read.parquet(path).createOrReplaceTempView(name)
  }
}