__author__ = 'Joerg Franke'
"""
This file contains update weights class includes the optimization functions for the network
"""

######                           Imports
########################################
import numpy as np
import theano
import theano.tensor as T



######    Nesterov momentum with RMSPROP
########################################
class nm_rmsprop:
    def __init__(self, rng,  weights,):

        cache_np = []
        for w in weights:
            cache_np.append(rng.uniform(0.0, 1.0, w.get_value().shape ))
        self.t_cache = []
        for c in cache_np:
            self.t_cache.append(theano.shared(value=c.astype(T.config.floatX)))


        velocity_np = []
        for w in weights:
            velocity_np.append(np.zeros(w.get_value().shape ))
        self.t_velocity = []
        for c in velocity_np:
            self.t_velocity.append(theano.shared(value=c.astype(T.config.floatX)))

    def fit(self, weights, o_error, tpo ):

        grades = T.grad(o_error ,weights)
        updates = []
        for c, v, w, g in zip(self.t_cache, self.t_velocity, weights,grades):
            gradient = g
            new_velocity = T.sub( T.mul(tpo["momentum"], v) , T.mul(tpo["learn_rate"], gradient) )
            new_cache = T.add( T.mul(tpo["decay_rate"] , c) , T.mul(T.sub( 1, tpo["decay_rate"]) , T.sqr(gradient)))
            new_weights = T.sub(T.add(w , new_velocity) , T.true_div( T.mul(gradient,tpo["learn_rate"]) , T.sqrt(T.add(new_cache,0.1**8))))
            updates.append((w, new_weights))
            updates.append((v, new_velocity))
            updates.append((c, new_cache))

        return updates


######                 Nesterov momentum
########################################
class nesterov_momentum:
    def __init__(self, rng,  weights):

        velocity_np = []
        for w in weights:
            velocity_np.append(np.zeros(w.get_value().shape ))
        self.t_velocity = []
        for c in velocity_np:
            self.t_velocity.append(theano.shared(value=c.astype(T.config.floatX)))

    def fit(self, weights, o_error, tpo):

        updates = []
        for v, w in zip(self.t_velocity, weights):
            gradient = T.grad(o_error ,w)
            new_velocity = tpo["momentum"] * v - tpo["learn_rate"] * gradient
            new_weights = w + new_velocity
            updates.append((w, new_weights))
            updates.append((v, new_velocity))
        return updates


######                           RMSPROP
########################################
"""
RMS PROP
http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf
"""
class rmsprop:
    def __init__(self, rng,  weights):
        cache_np = []
        for w in weights:
            cache_np.append(rng.uniform(0.0, 1.0, w.get_value().shape ))
        self.t_cache = []
        for c in cache_np:
            self.t_cache.append(theano.shared(value=c.astype(T.config.floatX)))

    def fit(self, weights, o_error, tpo):
        updates = []
        for c, w in zip(self.t_cache, weights):
            gradient = T.grad(o_error ,w)
            new_cache = tpo["decay_rate"] * c + ( 1- tpo["decay_rate"]) * T.sqr(gradient)
            new_weights = w - (gradient * tpo["learn_rate"]) / T.sqrt(new_cache + 0.1**8)
            updates.append((w, new_weights))
            updates.append((c, new_cache))

        return updates


######                          ADADELTA
########################################
class adadelta:
    def __init__(self, rng,  weights):
        ada_g = []
        for w in weights:
            ada_g.append(np.zeros(w.get_value().shape ))
        self.t_ada_g = []
        for c in ada_g:
            self.t_ada_g.append(theano.shared(value=c.astype(T.config.floatX)))

        ada_d = []
        for w in weights:
            ada_d.append(np.zeros(w.get_value().shape ))
        self.t_ada_d = []
        for c in ada_d:
            self.t_ada_d.append(theano.shared(value=c.astype(T.config.floatX)))

    def fit(self, weights, o_error, tpo):
        epsilon = 1e-6 #for numerical stability
        rho = 0.95
        updates = []
        for d, g, w in zip(self.t_ada_d, self.t_ada_g, weights):
            gradient = T.grad(o_error ,w)

            new_ada_g = rho * g + (1-rho) * T.sqr(gradient)

            delta_w = - T.sqrt(d + epsilon) * gradient / T.sqrt( new_ada_g + epsilon )

            new_ada_d = rho * d + (1-rho) * T.sqr(delta_w)

            new_weight = w + delta_w

            updates.append((w, new_weight))
            updates.append((g, new_ada_g))
            updates.append((d, new_ada_d))

        return updates


######                       Vanilla SGD
########################################
class sgd:
    def __init__(self, rng,  weights):
        pass


    def fit(self, weights, o_error, tpo):

        updates = []
        for w in weights:
            gradient = T.grad(o_error ,w)
            new_weights = w - (gradient * tpo["learn_rate"])
            updates.append((w, new_weights))

        return updates