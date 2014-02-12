% PROPAGATION_KERNEL calculates the propagation kernel between graphs.
%
% This function contains an implementation of the propagation graph
% kernel described in:
%
%   Neumann, M., Patricia, N., Garnett, R., and Kersting, K. Efficient
%   Graph Kernels by Randomization. (2012). Machine Learning And
%   Knowledge Discovery in Databases: European Conference, (ECML/PKDD
%   2012), pp. 378-392.
%
% This implementation supports (approximately) preserving any of the
% following distances between the feature vectors:
%
%   - \ell^1 or \ell^2 for arbitrary feature vectors
%   - the total variation or Hellinger distance for
%     distribution-valued features
%
% Depending on the chosen distance, the input features will be
% transformed appropriately.
%
% This implementation also supports arbitrary transformations to be
% used. See the transformations/ directory for example implementations
% (label diffusion and label propagation). Transformations must
% satisfy the following very general interface:
%
%   features = transformation(features);
%
% Where the input is an (n x d) set of feature vectors, one for
% each node, and the transformation function computes a new set of
% features given an old set.
%
% Usage:
%
%   K = propagation_kernel(features, graph_ind, transformation, ...
%                          num_iterations, varargin)
%
% Inputs:
%
%         features: an (n x d) matrix of feature vectors to hash. If
%                   either the total variation or Hellinger distances
%                   is chosen, each row of this matrix should sum to
%                   1.
%        graph_ind: an (n x 1) vector indicating graph membership.
%                   Each value of graph_ind should be a positive
%                   integer indicating the ordinal number of the graph
%                   the corresponding node belongs to. Given an
%                   adjacency matrix A, a suitable vector can be found
%                   via, e.g.:
%
%                     [~, graph_ind] = ...
%                         graphconncomp(A, 'directed', false);
%
%   transformation: a function handle to a feature transformation
%                   function satisfying the above-described API.
%   num_iterations: the number of iterations to use for the kernel
%                   computation.
%
% Optional inputs (specified as name/value pairs):
%
%             'w': the bin width to use during the hashing
%                  computation (default: 1e-4)
%      'distance': a string indicating the distance to approximately
%                  preserve; the following values are supported:
%
%                'l1': \ell^1
%                'l2': \ell^2
%                'tv': total variation distance (equivalent to \ell^1)
%         'hellinger': Hellinger distance
%
%                 The input is not case sensitive. See
%                 calculate_hashes for more information.
%                 (default: 'l1').
%
%   'base_kernel': a function handle to the base kernel to use. The
%                  kernel will be called as:
%
%                    K = base_kernel(counts),
%
%                  where counts is a (m x k) matrix, where m is the
%                  number of graphs and k is the number of unique
%                  hashes during this step of the computation.
%                  counts(i, j) contains the number of times hash j
%                  occurs in graph i. The default base kernel is the
%                  linear one:
%
%                    @(counts) (counts * counts');
%
% Outputs:
%
%   K: an (m x m) matrix containing the computed propagation kernel.
%
% See also CALCULATE_HASHES, LABEL_DIFFUSION, LABEL_PROPAGATION.

% Copyright (c) Roman Garnett, 2012--2014.

function K = propagation_kernel(features, graph_ind, transformation, ...
                                num_iterations, varargin)

  % parse optional inputs
  options = inputParser;

  % which distance to use
  options.addOptional('distance', 'l1', ...
                      @(x) ismember(lower(x), {'l1', 'l2', 'tv', 'hellinger'}));

  % width for hashing
  options.addOptional('w', 1e-4, ...
                      @(x) (isscalar(x) && (x > 0)));

  % base kernel for counts vector, default is linear kernel
  options.addOptional('base_kernel', ...
                      @(counts) (counts * counts'), ...
                      @(x) (isa(x, 'function_handle')));

  options.parse(varargin{:});
  options = options.Results;

  num_graphs = max(graph_ind);

  % initialize output
  K = zeros(num_graphs);

  iteration = 0;
  while (true)
    labels = calculate_hashes(features, options.distance, options.w);

    % aggregate counts on graphs
    counts = accumarray([graph_ind, labels], 1);

    % contribution specified by base kernel on count vectors
    K = K + options.base_kernel(counts);

    % avoid unnecessary transformation on last iteration
    if (iteration == num_iterations)
      break;
    end

    % apply transformation to features for next step
    features = transformation(features);

    iteration = iteration + 1;
  end

end