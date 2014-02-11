% Simple label diffusion transformation.

function features = label_diffusion(features, A)

  features = A * features;

end