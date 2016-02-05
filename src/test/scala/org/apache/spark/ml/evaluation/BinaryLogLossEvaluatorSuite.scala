/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.apache.spark.ml.evaluation

import org.apache.spark.SparkFunSuite
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.util.MLlibTestSparkContext

class BinaryLogLossEvaluatorSuite extends SparkFunSuite with MLlibTestSparkContext {

  test("should accept both vector and double raw prediction col") {
    val evaluator = new BinaryLogLossEvaluator()

    val df = sqlContext.createDataFrame(Seq(
      (1d, Vectors.dense(0.1, 0.9)),
      (0d, Vectors.dense(0.9, 0.2)),
      (0d, Vectors.dense(0.8, 0.2)),
      (1d, Vectors.dense(0.35, 0.65))
    )).toDF("label", "probability")
    assert(math.abs(evaluator.evaluate(df) - 0.2456) < 10e-4)
  }
}