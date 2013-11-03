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
package trysetnull.com.github.optim

// http://doc.akka.io/docs/akka/snapshot/scala/futures.html #Use Directly, #Functional Futures

import org.apache.commons.math3.random.{MersenneTwister, Well44497a}
import org.apache.commons.math3.util.FastMath

import org.slf4j.LoggerFactory

import scala.annotation.tailrec
import scala.concurrent.{future, blocking, Future, Await}
import scala.concurrent.ExecutionContext.Implicits.global
import scala.util.Success

/**
 * particle, x_s: Vector[Double]
 * particle length = m; j: 0 -> m-1
 *
 * x_s has already been evaluated.
 * x_p is known from the previous step.
 * v is a small constant.
 * 
 * x_s'[j] = G((x_s[j] + x_p[j])/2, FastMath.max(FastMath.abs(x_s[j] - x_p[j]), v))
 * 
 * Easy way to generate the gaussian distributed numbers that we need.
 */
sealed class NonCollapsingGaussian {
  private val rng = new MersenneTwister()

  /* Center a gaussian distribution between x_s and x_p */
  def sample(x_s: Double, x_p: Double, v: Double): Double = {
    FastMath.max(FastMath.abs(x_s - x_p), v) * rng.nextGaussian + (x_s + x_p)/2 
  }
}

/*
 * Particle swarm optimization.
 * @param bounds: (min, max) for each parameter.
 * @enforce: if true then the parameters are guranteed to lie within their respective bounds.
 * @swarmSize: The size of the swarm. 100 particles by default.
 */
class Swarm(swarmSize: Int, bounds: Seq[(Double, Double)], enforce: Boolean, var finished: Double => Boolean) {

  def this(bounds: Seq[(Double, Double)], enforce: Boolean) = this(100, bounds, enforce, x => true)

  val log = LoggerFactory.getLogger(getClass)

  def invoke(): Unit = {
    val futureParticles = Future.traverse((1 to swarmSize).toList)(x => createParticle)
    swarm(futureParticles)
  }

  // TODO: A recursive method with an accumulator, mutable list or stream.
  //@tailrec
  private def swarm(currentParticles: Future[List[(Seq[Double], Double)]]): Unit = {
    currentParticles onSuccess {
      case x => if (finished(x.flatMap(_._1).min)) return
    }
    
    val nextIteration = currentParticles flatMap {
      particles: List[(Seq[Double], Double)] => 
      {
	val (minimum, score) = particles.reduce((acc, next) => if (acc._2 < next._2) acc else next)
	log.info(score+":\t"+minimum)
	Future.traverse(particles)(p => updatePosition(p, minimum))
      }
    }
    
    nextIteration onSuccess {
      case _ => swarm(nextIteration)
    }
  }

  /* generates a particle */
  private def createParticle: Future[(Seq[Double], Double)] =  {
    future {
      // Perform a uniform sample on the bound.
      // Evaluate the parameter set returning both the particle and its fitness.
      // A particle will contain its own gaussian distribution instance.
      Thread.sleep(10)
      (List(1.9,1.8), 0.0)
    }
  }

  /* Updates a particle of the swarm using the currently found optimum.
   */
  private def updatePosition(particle: (Seq[Double], Double), optimum: Seq[Double]): Future[(Seq[Double], Double)] = {
    future {
      // Sample the NonCollapsingGaussian
      // checking that bound constraints are fulfilled.
      // Evalate the parameter set returning both particle and its fitness.
      Thread.sleep(10)
      (List(1.0, 2.0, 3.0), 4.0)
    }
  }
}
