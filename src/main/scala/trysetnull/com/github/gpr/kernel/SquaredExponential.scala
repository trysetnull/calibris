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
 * SquaredExponential.scala
 */
package trysetnull.com.github.gpr.kernel

import org.apache.commons.math3.analysis.differentiation.DerivativeStructure
import org.apache.commons.math3.util.{FastMath, Precision}

/**
 * The Squared exponential kernel function and its first derivative.
 * @param sigma_f signal standard deviation.
 * @param sigma_n noise standard deviation.
 * @param l length scale. 
 */
class SquaredExponential(sigma_f: Double, sigma_n: Double, l: Double) extends KernelBase {
  // DerivativeStructure
  // kernel function k has two free parmeters: k(x_p, x_q)
  // We want to make first-order derivative predictions which require us to be
  // able to differentiate the kernel to second order.
  val params = 2
  val order = 2

  // Kronecker delta
  val variance_n = FastMath.pow(sigma_n, 2)
  val S = 1E3*Precision.EPSILON // stability constant
  val stabilization = if (S > variance_n) S else 0.0
  val f_noise = this.kroneckerDelta(variance_n + stabilization)

  // Covariance function arguments
  val variance_f = FastMath.pow(sigma_f, 2)
  val l2 = FastMath.pow(l, 2)
  
  override def covariance(p: Int, q: Int): (Double, Double) => DerivativeStructure = {
    val noise = f_noise(p, q)
    //(x_p, x_q) => variance_f * FastMath.exp(-0.5 * FastMath.pow(x_p - x_q, 2)/l2) + noise
    (x_p_value, x_q_value) => {
      val x_p = new DerivativeStructure(params, order, 0, x_p_value)
      val x_q = new DerivativeStructure(params, order, 1, x_q_value)
      x_p.subtract(x_q).pow(2).divide(l2).multiply(-0.5).exp().multiply(variance_f).add(noise)
    }
  }
}
