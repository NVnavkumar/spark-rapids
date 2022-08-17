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

import org.apache.spark.sql.catalyst.plans.{LeftAnti, LeftSemi}
import org.apache.spark.sql.catalyst.plans.logical.Statistics
import org.apache.spark.sql.catalyst.plans.logical.statsEstimation.EstimationUtils
import org.apache.spark.sql.execution.{LeafExecNode, UnaryExecNode}
import org.apache.spark.sql.rapids.execution.GpuHashJoin

trait GpuPlanVisitor[T] {
  def visit(p: GpuExec): T = p match {
    case p: GpuHashJoin => visitJoin(p)
    case p: GpuExpandExec => visitExpand(p)
    case p: GpuHashAggregateExec => visitAggregate(p)
    case p: UnaryExecNode => visitUnaryNode(p)
    case p: GpuExec => default(p)
  }

  def default(p: GpuExec): T

  def visitAggregate(p: GpuHashAggregateExec): T

  def visitUnaryNode(p: UnaryExecNode): T

  def visitJoin(p: GpuHashJoin): T

  def visitExpand(p: GpuExpandExec): T
}

object GpuSizeInBytesOnlyStatsPlanVisitor extends GpuPlanVisitor[Statistics] {

  override def visitUnaryNode(p: UnaryExecNode): Statistics = {
    // There should be some overhead in Row object, the size should not be zero when there is
    // no columns, this help to prevent divide-by-zero error.
    // TODO: Use a GPU version of these 2 estimations
    val childRowSize = EstimationUtils.getSizePerRow(p.child.output)
    val outputRowSize = EstimationUtils.getSizePerRow(p.output)
    // Assume there will be the same number of rows as child has.
    var sizeInBytes = (p.child.stats.sizeInBytes * outputRowSize) / childRowSize
    if (sizeInBytes == 0) {
      // sizeInBytes can't be zero, or sizeInBytes of BinaryNode will also be zero
      // (product of children).
      sizeInBytes = 1
    }

    // Don't propagate rowCount and attributeStats, since they are not estimated here.
    Statistics(sizeInBytes = sizeInBytes) 
  }

  override def default(p: GpuExec): Statistics = p match {
    case p: LeafExecNode => p.computeStats()
    case _: GpuExec => 
      Statistics(sizeInBytes = p.children.map(_.stats.sizeInBytes).filter(_ > 0L).product)
  }

  override def visitAggregate(p: GpuHashAggregateExec): Statistics = {
    if (p.groupingExpressions.isEmpty) {
      Statistics(
        sizeInBytes = EstimationUtils.getOutputSize(p.output, outputRowCount = 1),
        rowCount = Some(1))
    } else {
      visitUnaryNode(p)
    }
  }

  override def visitJoin(p: GpuHashJoin): Statistics = {
    p.joinType match {
      case LeftAnti | LeftSemi =>
        // LeftSemi and LeftAnti won't ever be bigger than left
        p.left.stats
      case _ =>
        default(p)
    } 
  }

  override def visitExpand(p: GpuExpandExec): Statistics = {
    val sizeInBytes = visitUnaryNode(p).sizeInBytes * p.projections.length
    Statistics(sizeInBytes = sizeInBytes)
  }
}