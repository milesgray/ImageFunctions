import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.stats import kendalltau
import torch
is_cuda = torch.cuda.is_available()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def sinkhorn_unrolled(c, a, b, num_sink, lambd_sink):
	"""
	An implementation of a Sinkhorn layer with Automatic Differentiation (AD).
	The format of input parameters and outputs is equivalent to the 'Sinkhorn' module below.
	"""
	log_p = -c / lambd_sink
	log_a = torch.log(a).unsqueeze(dim=-1)
	log_b = torch.log(b).unsqueeze(dim=-2)
	for _ in range(num_sink):
		log_p = log_p - (torch.logsumexp(log_p, dim=-2, keepdim=True) - log_b)
		log_p = log_p - (torch.logsumexp(log_p, dim=-1, keepdim=True) - log_a)
	p = torch.exp(log_p)
	return p


class Sinkhorn(torch.autograd.Function):
	"""
	An implementation of a Sinkhorn layer with our custom backward module, based on implicit differentiation
	:param c: input cost matrix, size [*,m,n], where * are arbitrarily many batch dimensions
	:param a: first input marginal, size [*,m]
	:param b: second input marginal, size [*,n]
	:param num_sink: number of Sinkhorn iterations
	:param lambd_sink: entropy regularization weight
	:return: optimized soft permutation matrix
	"""
	@staticmethod
	def forward(ctx, c, a, b, num_sink, lambd_sink):
		log_p = -c / lambd_sink
		log_a = torch.log(a).unsqueeze(dim=-1)
		log_b = torch.log(b).unsqueeze(dim=-2)
		for _ in range(num_sink):
			log_p -= (torch.logsumexp(log_p, dim=-2, keepdim=True) - log_b)
			log_p -= (torch.logsumexp(log_p, dim=-1, keepdim=True) - log_a)
		p = torch.exp(log_p)

		ctx.save_for_backward(p, torch.sum(p, dim=-1), torch.sum(p, dim=-2))
		ctx.lambd_sink = lambd_sink
		return p

	@staticmethod
	def backward(ctx, grad_p):
		p, a, b = ctx.saved_tensors

		m, n = p.shape[-2:]
		batch_shape = list(p.shape[:-2])

		grad_p *= -1 / ctx.lambd_sink * p
		K = torch.cat((torch.cat((torch.diag_embed(a), p), dim=-1),
					   torch.cat((p.transpose(-2, -1), torch.diag_embed(b)), dim=-1)), dim=-2)[..., :-1, :-1]
		t = torch.cat((grad_p.sum(dim=-1), grad_p[..., :, :-1].sum(dim=-2)), dim=-1).unsqueeze(-1)
		grad_ab, _ = torch.solve(t, K)
		grad_a = grad_ab[..., :m, :]
		grad_b = torch.cat((grad_ab[..., m:, :], torch.zeros(batch_shape + [1, 1], device=device, dtype=torch.float32)), dim=-2)
		U = grad_a + grad_b.transpose(-2, -1)
		grad_p -= p * U
		grad_a = -ctx.lambd_sink * grad_a.squeeze(dim=-1)
		grad_b = -ctx.lambd_sink * grad_b.squeeze(dim=-1)
		return grad_p, grad_a, grad_b, None, None, None

def to_var(x):
	if is_cuda:
		x = x.cuda()
	return x

def my_sample_gumbel(shape, eps=1e-20):
	"""Samples arbitrary-shaped standard gumbel variables.
	Args:
	shape: list of integers
	eps: float, for numerical stability
	Returns:
	A sample of standard Gumbel random variables
	"""
	#Sample from Gumbel(0, 1)
	U = torch.rand(shape).float()
	return -torch.log(eps - torch.log(U + eps))

def simple_sinkhorn(MatrixA, n_iter = 20):
	#performing simple Sinkhorn iterations.

	for i in range(n_iter):
		MatrixA /= MatrixA.sum(dim=1, keepdim=True)
		MatrixA /= MatrixA.sum(dim=2, keepdim=True)
	return MatrixA

def sinkhorn(log_alpha, n_iters = 20):
	# torch version
	"""Performs incomplete Sinkhorn normalization to log_alpha.
	By a theorem by Sinkhorn and Knopp [1], a sufficiently well-behaved  matrix
	with positive entries can be turned into a doubly-stochastic matrix
	(i.e. its rows and columns add up to one) via the successive row and column
	normalization.
	-To ensure positivity, the effective input to sinkhorn has to be
	exp(log_alpha) (element wise).
	-However, for stability, sinkhorn works in the log-space. It is only at
	return time that entries are exponentiated.
	[1] Sinkhorn, Richard and Knopp, Paul.
	Concerning nonnegative matrices and doubly stochastic
	matrices. Pacific Journal of Mathematics, 1967
	Args:
	log_alpha: a 2D tensor of shape [N, N]
	n_iters: number of sinkhorn iterations (in practice, as little as 20
		iterations are needed to achieve decent convergence for N~100)
	Returns:
	A 3D tensor of close-to-doubly-stochastic matrices (2D tensors are
		converted to 3D tensors with batch_size equals to 1)
	"""
	n = log_alpha.size()[1]
	log_alpha = log_alpha.view(-1, n, n)

	for i in range(n_iters):
		log_alpha = log_alpha - (torch.logsumexp(log_alpha, dim=2, keepdim=True)).view(-1, n, 1)
		log_alpha = log_alpha - (torch.logsumexp(log_alpha, dim=1, keepdim=True)).view(-1, 1, n)
	return torch.exp(log_alpha)

def gumbel_sinkhorn(log_alpha, temp=1.0, n_samples=1, noise_factor=1.0, n_iters=20, squeeze=True):
	"""Random doubly-stochastic matrices via gumbel noise.
	In the zero-temperature limit sinkhorn(log_alpha/temp) approaches
	a permutation matrix. Therefore, for low temperatures this method can be
	seen as an approximate sampling of permutation matrices, where the
	distribution is parameterized by the matrix log_alpha
	The deterministic case (noise_factor=0) is also interesting: it can be
	shown that lim t->0 sinkhorn(log_alpha/t) = M, where M is a
	permutation matrix, the solution of the
	matching problem M=arg max_M sum_i,j log_alpha_i,j M_i,j.
	Therefore, the deterministic limit case of gumbel_sinkhorn can be seen
	as approximate solving of a matching problem, otherwise solved via the
	Hungarian algorithm.
	Warning: the convergence holds true in the limit case n_iters = infty.
	Unfortunately, in practice n_iter is finite which can lead to numerical
	instabilities, mostly if temp is very low. Those manifest as
	pseudo-convergence or some row-columns to fractional entries (e.g.
	a row having two entries with 0.5, instead of a single 1.0)
	To minimize those effects, try increasing n_iter for decreased temp.
	On the other hand, too-low temperature usually lead to high-variance in
	gradients, so better not choose too low temperatures.
	Args:
	log_alpha: 2D tensor (a matrix of shape [N, N])
		or 3D tensor (a batch of matrices of shape = [batch_size, N, N])
	temp: temperature parameter, a float.
	n_samples: number of samples
	noise_factor: scaling factor for the gumbel samples. Mostly to explore
		different degrees of randomness (and the absence of randomness, with
		noise_factor=0)
	n_iters: number of sinkhorn iterations. Should be chosen carefully, in
		inverse correspondence with temp to avoid numerical instabilities.
	squeeze: a boolean, if True and there is a single sample, the output will
		remain being a 3D tensor.
	Returns:
			sink: a 4D tensor of [batch_size, n_samples, N, N] i.e.
				batch_size *n_samples doubly-stochastic matrices. If n_samples = 1 and
				squeeze = True then the output is 3D.
			log_alpha_w_noise: a 4D tensor of [batch_size, n_samples, N, N] of
				noisy samples of log_alpha, divided by the temperature parameter. Ifmy_invert_listperm
				n_samples = 1 then the output is 3D.
	"""
	n = log_alpha.size()[1]
	log_alpha = log_alpha.view(-1, n, n)
	batch_size = log_alpha.size()[0]

	log_alpha_w_noise = log_alpha.repeat(n_samples, 1, 1)

	if noise_factor == 0:
		noise = 0.0
	else:
		noise = to_var(my_sample_gumbel([n_samples*batch_size, n, n])*noise_factor)

	log_alpha_w_noise = log_alpha_w_noise + noise
	log_alpha_w_noise = log_alpha_w_noise / temp

	my_log_alpha_w_noise = log_alpha_w_noise.clone()

	sink = sinkhorn(my_log_alpha_w_noise)
	if n_samples > 1 or squeeze is False:
		sink = sink.view(n_samples, batch_size, n, n)
		sink = torch.transpose(sink, 1, 0)
		log_alpha_w_noise = log_alpha_w_noise.view(n_samples, batch_size, n, n)
		log_alpha_w_noise = torch.transpose(log_alpha_w_noise, 1, 0)
	return sink, log_alpha_w_noise

def sample_uniform_and_order(n_lists, n_numbers, prob_inc):
	"""Samples uniform random numbers, return sorted lists and the indices of their original values
	Returns a 2-D tensor of n_lists lists of n_numbers sorted numbers in the [0,1]
	interval, each of them having n_numbers elements.
	Lists are increasing with probability prob_inc.
	It does so by first sampling uniform random numbers, and then sorting them.
	Therefore, sorted numbers follow the distribution of the order statistics of
	a uniform distribution.
	It also returns the random numbers and the lists of permutations p such
	p(sorted) = random.
	Notice that if one ones to build sorted numbers in different intervals, one
	might just want to re-scaled this canonical form.
	Args:
	n_lists: An int,the number of lists to be sorted.
	n_numbers: An int, the number of elements in the permutation.
	prob_inc: A float, the probability that a list of numbers will be sorted in
	increasing order.
	Returns:
	ordered: a 2-D float tensor with shape = [n_list, n_numbers] of sorted lists
		of numbers.
	random: a 2-D float tensor with shape = [n_list, n_numbers] of uniform random
		numbers.
	permutations: a 2-D int tensor with shape = [n_list, n_numbers], row i
		satisfies ordered[i, permutations[i]) = random[i,:].
	"""
	# sample n_lists samples from Bernoulli with probability of prob_inc
	my_bern = torch.distributions.Bernoulli(torch.tensor([prob_inc])).sample([n_lists])

	sign = -1*((my_bern * 2) -torch.ones([n_lists,1]))
	sign = sign.type(torch.float32)
	random =(torch.empty(n_lists, n_numbers).uniform_(0, 1))
	random =random.type(torch.float32)

	# my change
	#random_with_sign = random * sign
	#Finds sorted values and indices of the k largest entries for the last dimension.
	#sorted – controls whether to return the elements in sorted order

	#ordered, permutations = torch.topk(random_with_sign, k = n_numbers, sorted = True)
	# my change
	ordered, permutations = torch.sort(random, descending=True)
	#my change
	#ordered = ordered * sign
	return (ordered, random, permutations)

def sample_permutations(n_permutations, n_objects):
	"""Samples a batch permutations from the uniform distribution.
	Returns a sample of n_permutations permutations of n_objects indices.
	Permutations are assumed to be represented as lists of integers
	(see 'listperm2matperm' and 'matperm2listperm' for conversion to alternative
	matricial representation). It does so by sampling from a continuous
	distribution and then ranking the elements. By symmetry, the resulting
	distribution over permutations must be uniform.
	Args:
	n_permutations: An int, the number of permutations to sample.
	n_objects: An int, the number of elements in the permutation.
		the embedding sources.
	Returns:
	A 2D integer tensor with shape [n_permutations, n_objects], where each
		row is a permutation of range(n_objects)
	"""
	random_pre_perm = torch.empty(n_permutations, n_objects).uniform_(0, 1)
	_, permutations = torch.topk(random_pre_perm, k = n_objects)
	return permutations

def permute_batch_split(batch_split, permutations):
	"""Scrambles a batch of objects according to permutations.
	It takes a 3D tensor [batch_size, n_objects, object_size]
	and permutes items in axis=1 according to the 2D integer tensor
	permutations, (with shape [batch_size, n_objects]) a list of permutations
	expressed as lists. For many dimensional-objects (e.g. images), objects have
	to be flattened so they will respect the 3D format, i.e. tf.reshape(
	batch_split, [batch_size, n_objects, -1])
	Args:
	batch_split: 3D tensor with shape = [batch_size, n_objects, object_size] of
		splitted objects
	permutations: a 2D integer tensor with shape = [batch_size, n_objects] of
		permutations, so that permutations[n] is a permutation of range(n_objects)
	Returns:
	A 3D tensor perm_batch_split with the same shape as batch_split,
		so that perm_batch_split[n, j,:] = batch_split[n, perm[n,j],:]
	"""
	batch_size= permutations.size()[0]
	n_objects = permutations.size()[1]

	permutations = permutations.view(batch_size, n_objects, -1)
	perm_batch_split = torch.gather(batch_split, 1, permutations)
	return perm_batch_split


def listperm2matperm(listperm):
	"""Converts a batch of permutations to its matricial form.
	Args:
	listperm: 2D tensor of permutations of shape [batch_size, n_objects] so that
		listperm[n] is a permutation of range(n_objects).
	Returns:
	a 3D tensor of permutations matperm of
		shape = [batch_size, n_objects, n_objects] so that matperm[n, :, :] is a
		permutation of the identity matrix, with matperm[n, i, listperm[n,i]] = 1
	"""
	n_objects = listperm.size()[1]
	eye = np.eye(n_objects)[listperm]
	eye= torch.tensor(eye, dtype=torch.int32)
	return eye

def matperm2listperm(matperm):
	"""Converts a batch of permutations to its enumeration (list) form.
	Args:
	matperm: a 3D tensor of permutations of
		shape = [batch_size, n_objects, n_objects] so that matperm[n, :, :] is a
		permutation of the identity matrix. If the input is 2D, it is reshaped
		to 3D with batch_size = 1.
	dtype: output_type (int32, int64)
	Returns:
	A 2D tensor of permutations listperm, where listperm[n,i]
	is the index of the only non-zero entry in matperm[n, i, :]
	"""
	batch_size = matperm.size()[0]
	n_objects = matperm.size()[1]
	matperm = matperm.view(-1, n_objects, n_objects)

	#argmax is the index location of each maximum value found(argmax)
	_, argmax = torch.max(matperm, dim=2, keepdim= True)
	argmax = argmax.view(batch_size, n_objects)
	return argmax

def invert_listperm(listperm):
	"""Inverts a batch of permutations.
	Args:
	listperm: a 2D integer tensor of permutations listperm of
		shape = [batch_size, n_objects] so that listperm[n] is a permutation of
		range(n_objects)
	Returns:
	A 2D tensor of permutations listperm, where listperm[n,i]
	is the index of the only non-zero entry in matperm[n, i, :]
	"""
	return matperm2listperm(torch.transpose(listperm2matperm(listperm), 1, 2))

def matching(matrix_batch):
	"""Solves a matching problem for a batch of matrices.
	This is a wrapper for the scipy.optimize.linear_sum_assignment function. It
	solves the optimization problem max_P sum_i,j M_i,j P_i,j with P a
	permutation matrix. Notice the negative sign; the reason, the original
	function solves a minimization problem
	Args:
		matrix_batch: A 3D tensor (a batch of matrices) with
			shape = [batch_size, N, N]. If 2D, the input is reshaped to 3D with
			batch_size = 1.
	Returns:
		listperms, a 2D integer tensor of permutations with shape [batch_size, N]
			so that listperms[n, :] is the permutation of range(N) that solves the
			problem  max_P sum_i,j M_i,j P_i,j with M = matrix_batch[n, :, :].
	"""

	def hungarian(x):
		if x.ndim == 2:
			x = np.reshape(x, [1, x.shape[0], x.shape[1]])
		sol = np.zeros((x.shape[0], x.shape[1]), dtype=np.int32)
		for i in range(x.shape[0]):
			sol[i, :] = linear_sum_assignment(-x[i, :])[1].astype(np.int32)
		return sol

	listperms = hungarian(matrix_batch.detach().cpu().numpy())
	listperms = torch.from_numpy(listperms)
	return listperms

def kendall_tau(batch_perm1, batch_perm2):
	"""Wraps scipy.stats kendalltau function.
	Args:
		batch_perm1: A 2D tensor (a batch of matrices) with
			shape = [batch_size, N]
		batch_perm2: same as batch_perm1
	Returns:
		A list of Kendall distances between each of the elements of the batch.
	"""

	def kendalltau_batch(x, y):
		if x.ndim == 1:
			x = np.reshape(x, [1, x.shape[0]])
		if y.ndim == 1:
			y = np.reshape(y, [1, y.shape[0]])
		kendall = np.zeros((x.shape[0], 1), dtype=np.float32)
		for i in range(x.shape[0]):
			kendall[i, :] = kendalltau(x[i, :], y[i, :])[0]
		return kendall

	listkendall = kendalltau_batch(batch_perm1.cpu().numpy(), batch_perm2.cpu().numpy())
	listkendall = torch.from_numpy(listkendall)
	return listkendall