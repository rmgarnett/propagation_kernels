function K = propagation_kernel(A, graph_ind, labels, train_ind, ...
                                max_height, w, use_cauchy, use_pushback, ...
                                take_sum)

  num_nodes   = size(A, 1);
  num_graphs  = max(graph_ind);
  num_classes = max(labels);
  num_train   = numel(train_ind);

  % calculate and store the rows of the probability matrix
  % corresponding to the training nodes
  train_rows = accumarray([(1:num_train)', labels(train_ind)], 1, ...
                          [num_train, num_classes]);

  % initialize the probability matrix, using a uniform distribution
  % on the unlabeled nodes
  probabilities = (1 / num_classes) * ones(num_nodes, num_classes);
  probabilities(train_ind, :) = train_rows;

  % initialize output array
  if (take_sum)
    K = zeros(num_graphs);
  else
    K = zeros(num_graphs, num_graphs, max_height + 1);
  end

  height = 0;
  while (true)

    % calculate the kernel contribution at this height
    contribution = propagation_kernel_contribution(probabilities, ...
            graph_ind, w, use_cauchy);
    contribution = contribution + contribution' - diag(diag(contribution));

    if (take_sum)
      K = K + contribution;
    else
      K(:, :, height + 1) = contribution;
    end

    % avoid updating probabilities after the last contribution
    if (height == max_height)
      break;
    end

    % update probabilities
    probabilities = A * probabilities;

    % push back labels if desired
    if (use_pushback)
      probabilities(train_ind, :) = train_rows;
    end

    height = height + 1;
  end
end