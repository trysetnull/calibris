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
 * GaussianProcessRegression.scala
 */
package trysetnull.com.github.gpr

import org.apache.commons.math3.analysis.differentiation.DerivativeStructure
import org.apache.commons.math3.analysis.UnivariateFunction
import org.apache.commons.math3.distribution.MultivariateNormalDistribution
import org.apache.commons.math3.linear.{ArrayRealVector, CholeskyDecomposition, DecompositionSolver, LUDecomposition, MatrixUtils, RealMatrix, RealVector}
import org.apache.commons.math3.util.{FastMath, Precision}

import scala.util.control.Exception._

import trysetnull.com.github.gpr.kernel.KernelBase

/*
 * Performs Gaussian Process regression.
 * @param xy is a vector of (x, y) data points on which to perform the regression.
 * @param kernel is used to calculate the values of the covariance matrix.
 */
object GaussianProcessRegression {
  def apply(xy: Seq[(Double, Double)], kernel: KernelBase): Option[GaussianProcessRegression] = {
    val n = xy.length
    val K = MatrixUtils.createRealMatrix(n, n)
    for (p <- 0 until n; q <- p until n) {
      val cov = kernel.covariance(p, q)(xy(p)._1, xy(q)._1).getValue
      K.setEntry(p, q, cov)
      K.setEntry(q, p, cov) // the matrix is symmetric
    }

    for {
      chol <- allCatch opt new CholeskyDecomposition(K)
      l_solver <- allCatch opt new LUDecomposition(chol.getL()).getSolver()
      lt_solver <- allCatch opt new LUDecomposition(chol.getLT()).getSolver()
      
      y_obs_mean = xy.foldLeft(0.0)(_ + _._2)/n
      y = {
	val vector = new ArrayRealVector(n)
	var i = 0
	for (t <- xy) { vector.setEntry(i, t._2 - y_obs_mean); i += 1 }
	vector
      }
      
      // alpha := L'\(L\y)
      alpha  <- allCatch opt lt_solver.solve(l_solver.solve(y))
      l = chol.getL()
      sumLogL_ii = 0 until n map (i => FastMath.log(l.getEntry(i, i))) reduceLeft (_+_)
      // log p(y|X) := -1/2 * y' * alpha - SUM_i log L_ii - n/2 log 2*PI
      logpyx = -0.5*y.dotProduct(alpha) - sumLogL_ii - n*0.5*FastMath.log(2*FastMath.PI)
    } yield new GaussianProcessRegression(xy, kernel, l_solver, alpha, y_obs_mean, logpyx)
  }
}

class GaussianProcessRegression private (
  private val xy: Seq[(Double, Double)],
  private val kernel: KernelBase,
  private val l_solver: DecompositionSolver,
  private val alpha: RealVector,
  private val y_obs_mean: Double,
  val logMarginalLikelihood: Double
) {
  
  /** Predictions from the Gaussian Process Regression for a sequence of inputs x_*. */
  def predict(x_* : Seq[Double]): Option[(RealVector, RealMatrix)] = {
    predict(x_*, FUNCTION)
  }
  
  /** Predictions from the Gaussian Process Regression for a sequence of inputs x_*.
   *  Either the function value or its first derivative, specified by mode.
   *  Option will be empty if the internal matrix decomposition is singular. */
  def predict(x_* : Seq[Double], mode: PredictionMode): Option[(RealVector, RealMatrix)] = {
    val (k_*, k_**, f_mean) = mode match {
      case FUNCTION => (
	generateK_*(x_*, (x_p: Double, x_q: Double) => {
	  // noise free
	  kernel.covariance(0, 1)(x_p, x_q).getValue
	}),
	generateK_**(x_*, (p: Int, q: Int) => (x_p: Double, x_q: Double) => {
	  kernel.covariance(p, q)(x_p, x_q).getValue
	}),
	new FunctionClosure(mu => mu + y_obs_mean)
      )
      case DERIVATIVE => (
	generateK_*(x_*, (x_p: Double, x_q: Double) => {
	  // noise free value of dxdx*
	  kernel.covariance(0, 1)(x_p, x_q).getPartialDerivative(1, 0)
	}),
	generateK_**(x_*, (p: Int, q: Int) => (x_p: Double, x_q: Double) => {
	  kernel.covariance(p, q)(x_p, x_q).getPartialDerivative(1, 1)
	}),
	new FunctionClosure(mu => mu)
      )
    }
    
    for {
      // v := L\k_*'
      v <- allCatch opt l_solver.solve(k_*.transpose())
      // mean := k_* * alpha
      mean = k_*.operate(alpha).map(f_mean)
      // V|f_*| := k(x_*, x_*) - v'v
      covariance = k_**.subtract(v.transpose().multiply(v))
    } yield (mean, covariance)
  }

  /** The arguments can be generated with the #predict method within this class. */
  def sample(means: RealVector, covariances: RealMatrix, n: Int): IndexedSeq[Array[Double]] = {    
    val mvnorm = new MultivariateNormalDistribution(means.toArray, covariances.getData)
    return for (i <- 0 until n) yield {
      mvnorm.sample()
    }
  }
      
  /** Generates the K_* matrix from a sequence of inputs x_* and a kernel function k.
   * The training points are used implicitly. */
  private def generateK_*(x_* : Seq[Double], k: (Double, Double) => Double): RealMatrix = {    
    val k_* = MatrixUtils.createRealMatrix(x_*.length, xy.length)
    for (p <- 0 until k_*.getRowDimension; q <- 0 until k_*.getColumnDimension) {
      val cov = k(x_*(p), xy(q)._1)
      k_*.setEntry(p, q, cov)
    }
    k_*
  }
  
  // TODO: automatic calculation of the ridge.
  /** Generates the K_** matrix from a sequence of inputs x_* and a kernel function k. */
  private def generateK_**(x_* : Seq[Double], k: (Int, Int) => (Double, Double) => Double): RealMatrix = {
    val f_stabilization = kernel.kroneckerDelta(1E6*Precision.EPSILON)
    
    val k_** = MatrixUtils.createRealMatrix(x_*.length, x_*.length)
    for (p <- 0 until k_**.getRowDimension; q <- p until k_**.getRowDimension) {
      val cov = k(p, q)(x_*(p), x_*(q)) + f_stabilization(p, q)
      k_**.setEntry(p, q, cov)
      k_**.setEntry(q, p, cov) // the matrix is symmetric
    }
    k_**
  }
}

sealed trait PredictionMode
case object FUNCTION extends PredictionMode
case object DERIVATIVE extends PredictionMode

sealed class FunctionClosure(f: Double => Double) extends UnivariateFunction {
  def value(x: Double): Double = {
    f(x)
  }
}
