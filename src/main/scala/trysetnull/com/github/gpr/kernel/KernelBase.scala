/**
 * Copyright 2013 Tully Ernst
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
 *
 * KernelBase.scala
 */
package trysetnull.com.github.gpr.kernel

import org.apache.commons.math3.analysis.differentiation.DerivativeStructure

/** An abstract class for defining kernel functions to be used for GP regression. */
abstract class KernelBase {

  /** Returns a function that returns the noise * kronecker delta
   * the return value of which is intended to be added to a covariance matrix.
   * @param noise is typically the noise variance, i.e. sigma_n^2.
   * @param row index of the matrix
   * @param column index of the matrix
   */
  final def kroneckerDelta(noise: Double): (Int, Int) => Double = {
    return (p, q) => if (p == q) noise else 0.0
  }

  /** Defines the covariance function as a kernel. */
  def covariance(p: Int, q: Int): (Double, Double) => DerivativeStructure

}
