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
 * GaussianProcessRegressionSpec.scala
 */

import org.apache.commons.math3.analysis.differentiation.DerivativeStructure
import org.apache.commons.math3.linear.RealMatrix
import org.apache.commons.math3.stat.descriptive.SummaryStatistics
import org.apache.commons.math3.util.{FastMath, Precision}
import org.specs2.Specification
import org.specs2.specification.{Outside, Scope}

import trysetnull.com.github.gpr.DERIVATIVE
import trysetnull.com.github.gpr.FUNCTION
import trysetnull.com.github.gpr.GaussianProcessRegression
import trysetnull.com.github.gpr.kernel.KernelBase
import trysetnull.com.github.gpr.kernel.SquaredExponential

object TestData extends Outside[GaussianProcessRegression] with Scope {
  val sigma_f = 0.7
  val sigma_n = 0.0027
  val lengthScale = 1.0
  val xy = Seq((-4.0, -2.0), (-3.0, 0.0), (-1.0, 1.0), (0.0, 2.0), (2.0, -1.0))
  val kernel = new SquaredExponential(sigma_f, sigma_n, lengthScale)
  def outside: GaussianProcessRegression = GaussianProcessRegression(xy, kernel).get
  def apply(xy: Seq[(Double, Double)], kernel: KernelBase): GaussianProcessRegression = GaussianProcessRegression(xy, kernel).get
}

class GaussianProcessRegressionApproval extends Specification { def is =
  """
  Tests that the 'GaussianProcessRegression' implementation can produce a figure similar to [Figure 2.4, pp 18] of the Machine Learning book:
  C. E. Rasmussen & C. K. I. Williams, Gaussian Processes for Machine Learning, the MIT Press, 2006
  ISBN 026218253X
  www.GaussianProcess.org/gpml
  """ ^
  "The square exponential kernel returns values between 0 and "+(FastMath.pow(TestData.sigma_f, 2)+FastMath.pow(TestData.sigma_n, 2)) ! squaredExponential ^
  "The GaussianProcessRegression can be sampled." ! TestData(sampleVerification) ^
  "The GaussianProcessRegression can make predictions of the first derivative which are similar to the estimated slope via sampling." ! TestData(derivativePredictionVerification) ^
  "The GaussianProcessRegression can make predictions of the first derivative." ! TestData(derivativePredictionVerification2)

  def squaredExponential = {
    val sampleUniform: (Double, Double) => Double = {
      (lower, upper) => { val u = FastMath.random(); lower*(1.0-u) + upper*u }
    }
   
    // A mock of real x data points.
    val m = 100
    val x = (0 until m map (_ + sampleUniform(0.2, 1.0))).toVector
    
    val covariances = for {
      p <- 0 until m
      q <- p until m
      cov = TestData.kernel.covariance(p, q)(x(p), x(q)).getValue
    } yield cov
    
    forall(covariances) ((_: Double) must beBetween(0.0, FastMath.pow(TestData.sigma_f, 2) + FastMath.pow(TestData.sigma_n, 2)))
  }

  def sampleVerification(gpr: GaussianProcessRegression) = {
    val logpyX = gpr.logMarginalLikelihood

    val h = 0.125
    val upper = 5
    val lower = -5
    
    val k_star = 0 to ((upper - lower)/h).toInt map (_*h+lower)
    val (mu, cov) = gpr.predict(k_star.toSeq/*, DERIVATIVE */).get
    val variance = 0 until cov.getRowDimension map (i => cov.getEntry(i, i))
    val f_95int = (mu.toArray, variance).zipped.map((m, v) => m + 2*FastMath.sqrt(v))++
    (mu.toArray.reverse, variance.reverse).zipped.map((m, v) => m - 2*FastMath.sqrt(v))
    
    val sample = gpr.sample(mu, cov, 3)
    val sampleStr = sample.zipWithIndex.foldLeft("") { (acc, t) => acc + "s"+(t._2+1)+" = ["+t._1.mkString("; ")+"];\n" }

    // n x n positive definite symmetric matrix
    //println("COV = "+cov)

    // Create Octave plotting commands.
    /*
    println("x = ["+TestData.xy.map(t => t._1).mkString("; ")+"];")
    println("y = ["+TestData.xy.map(t => t._2).mkString("; ")+"];")
    println("z = ["+ k_star.mkString("; ")+"];")
    println("m = ["+ mu.toArray.mkString("; ")+"];")
    println("f = ["+f_95int.mkString("; ")+"];")
    println(sampleStr)
    println("fill([z; flipdim(z,1)], f, [7 7 7]/8);")
    println("hold on; plot(z, m); plot(z, s1, \"1\"); plot(z, s2, \"1\"); plot(z, s3, \"1\"); plot(x, y, '+');")
    */
    logpyX must be ~(-14.9 +/- 1E-1)
  }

  def derivativePredictionVerification(gpr: GaussianProcessRegression) = {
    // 5-point stencil for slope estimation.
    val fivePointStencil: (Double, Double, Double => Double) => Double = {
      (x, h, fx) => (-fx(x+2*h)+8*fx(x+h)-8*fx(x-h)+fx(x-2*h))/(12*h)
    }

    // Samples for a five point stencil around x.
    val h = 0.1
    val x = -3.6
    val k_star = Seq(x-2*h, x-h, x+h, x+2*h)
    val idxmap: Map[Double, Int] = Map(
      x-2*h -> 0, x-h -> 1, x+h -> 2, x+2*h -> 3
    )
    val (mu, cov) = gpr.predict(k_star.toSeq).get
    val samples = gpr.sample(mu, cov, 1E5.toInt).map(fx => {
      fivePointStencil(x, h, x_obs => fx(idxmap(x_obs)))
    })
    
    val summary = new SummaryStatistics()
    for (x <- samples) {
      summary.addValue(x)
    }
    
    /*
    println("Points of the five point stencil:\n"+k_star)
    println("Map to the index of the samples:\n"+idxmap)
    println("Estimated slope for samples for f(x)' mean and variance at x = "+x+":\n"+summary)
    */
    
    val (mu_dydx, cov_dydx) = gpr.predict(Seq(x), DERIVATIVE).get
    val variance = 0 until cov_dydx.getRowDimension map (i => cov_dydx.getEntry(i, i))
    //println("Predict for f(x)' mean: "+mu_dot+" and variance: "+variance)

    {mu_dydx.getEntry(0) must be ~(summary.getMean +/- 1E-3) updateMessage(mesg => "Mean: "+mesg)} and
    {variance(0) must be ~(summary.getVariance +/- 1E-3) updateMessage(mesg => "Variance: "+mesg)}
  }

  def derivativePredictionVerification2(gpr: GaussianProcessRegression) = {
    val h = 0.1
    val upper = 5
    val lower = -5
    val x_star = 0 to ((upper - lower)/h).toInt map (_*h+lower)

    // shinks the range domain of a slope between -1.0 and 1.0.
    val transformSlope: Double => Double = {
      m => m/FastMath.sqrt(1+FastMath.pow(m, 2))
    }

    val (mu, cov) = gpr.predict(x_star, DERIVATIVE).get
    val samples = gpr.sample(mu, cov, 5).flatMap(x => x)
    
    forall(samples map(x => transformSlope(x))) ((_: Double) must beBetween(-1.0, 1.0))
  }
}
