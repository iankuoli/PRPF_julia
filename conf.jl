function train_filepath(data_name::String, env::Int64)
  if env == 1
    #
    # Macbook in my house
    #
    if data_name == "Lastfm1K"
      # Last.fm1K
      training_path = "/Users/iankuoli/Dataset/LastFm1K_train.csv";
      testing_path = "/Users/iankuoli/Dataset/LastFm1K_test.csv";
      validation_path = "/Users/iankuoli/Dataset/LastFm1K_valid.csv";

    elseif data_name == "Lastfm2K"
      # Last.fm2K
      training_path = "/Users/iankuoli/Dataset/LastFm_train.csv";
      testing_path = "/Users/iankuoli/Dataset/LastFm_test.csv";
      validation_path = "/Users/iankuoli/Dataset/LastFm_valid.csv";

    elseif data_name == "Lastfm360K"
      # Last.fm360K
      training_path = "/Users/iankuoli/Dataset/LastFm360K_train.csv";
      testing_path = "/Users/iankuoli/Dataset/LastFm360K_test.csv";
      validation_path = "/Users/iankuoli/Dataset/LastFm360K_valid.csv";

    elseif data_name == "MovieLens100K"
      # MovieLens100K
      training_path = "/Users/iankuoli/Dataset/MovieLens100K_train.csv";
      testing_path = "/Users/iankuoli/Dataset/MovieLens100K_test.csv";
      validation_path = "/Users/iankuoli/Dataset/MovieLens100K_valid.csv";

    elseif data_name == "MovieLens1M"
      # MovieLens1M
      training_path = "/Users/iankuoli/Dataset/MovieLens1M_train.csv";
      testing_path = "/Users/iankuoli/Dataset/MovieLens1M_test.csv";
      validation_path = "/Users/iankuoli/Dataset/MovieLens1M_valid.csv";

    elseif data_name == "SmallToy"
      # SmallToy
      training_path = "/Users/iankuoli/Dataset/SmallToy_train.csv";
      testing_path = "/Users/iankuoli/Dataset/SmallToy_test.csv";
      validation_path = "/Users/iankuoli/Dataset/SmallToy_valid.csv";

    end
  elseif env == 2
    #
    # CentOS in my office
    #
    if data_name == "Lastfm1K"
      # Last.fm1K
      training_path = "/home/ian/Dataset/LastFm1K_train.csv";
      testing_path = "/home/ian/Dataset/LastFm1K_test.csv";
      validation_path = "/home/ian/Dataset/LastFm1K_valid.csv";

    elseif data_name == "Lastfm2K"
      # Last.fm2K
      training_path = "/home/ian/Dataset/LastFm_train.csv";
      testing_path = "/home/ian/Dataset/LastFm_test.csv";
      validation_path = "/home/ian/Dataset/LastFm_valid.csv";

    elseif data_name == "Lastfm360K"
      # Last.fm360K
      training_path = "/home/ian/Dataset/LastFm360K_train.csv";
      testing_path = "/home/ian/Dataset/LastFm360K_test.csv";
      validation_path = "/home/ian/Dataset/LastFm360K_valid.csv";

    elseif data_name == "MovieLens100K"
      # MovieLens100K
      training_path = "/home/ian/Dataset/MovieLens100K_train.csv";
      testing_path = "/home/ian/Dataset/MovieLens100K_test.csv";
      validation_path = "/home/ian/Dataset/MovieLens100K_valid.csv";

    elseif data_name == "MovieLens1M"
      # MovieLens1M
      training_path = "/home/ian/Dataset/MovieLens1M_train.csv";
      testing_path = "/home/ian/Dataset/MovieLens1M_test.csv";
      validation_path = "/home/ian/Dataset/MovieLens1M_valid.csv";

    elseif data_name == "SmallToy"
      # SmallToy
      training_path = "/home/ian/Dataset/SmallToy_train.csv";
      testing_path = "/home/ian/Dataset/SmallToy_test.csv";
      validation_path = "/home/ian/Dataset/SmallToy_valid.csv";
    end
  elseif env == 3
    #
    # CentOS in my GPU server
    #
    if data_name == "Lastfm1K"
      # Last.fm1K
      training_path = "/home/csist/Dataset/LastFm1K_train.csv";
      testing_path = "/home/csist/Dataset/LastFm1K_test.csv";
      validation_path = "/home/csist/Dataset/LastFm1K_valid.csv";

    elseif data_name == "Lastfm2K"
      # Last.fm2K
      training_path = "/home/csist/Dataset/LastFm_train.csv";
      testing_path = "/home/csist/Dataset/LastFm_test.csv";
      validation_path = "/home/csist/Dataset/LastFm_valid.csv";

    elseif data_name == "Lastfm360K"
      # Last.fm360K
      training_path = "/Users/csist/Dataset/LastFm360K_train.csv";
      testing_path = "/Users/csist/Dataset/LastFm360K_test.csv";
      validation_path = "/Users/csist/Dataset/LastFm360K_valid.csv";

    elseif data_name == "MovieLens100K"
      # MovieLens100K
      training_path = "/home/csist/Dataset/MovieLens100K_train.csv";
      testing_path = "/home/csist/Dataset/MovieLens100K_test.csv";
      validation_path = "/home/csist/Dataset/MovieLens100K_valid.csv";

    elseif data_name == "MovieLens1M"
      # MovieLens1M
      training_path = "/home/csist/Dataset/MovieLens1M_train.csv";
      testing_path = "/home/csist/Dataset/MovieLens1M_test.csv";
      validation_path = "/home/csist/Dataset/MovieLens1M_valid.csv";

    elseif data_name == "SmallToy"
      # SmallToy
      training_path = "/home/csist/Dataset/SmallToy_train.csv";
      testing_path = "/home/csist/Dataset/SmallToy_test.csv";
      validation_path = "/home/csist/Dataset/SmallToy_valid.csv";
    end
  end
  return training_path, testing_path, validation_path
end

function train_setting(data_name::String, model::String)
  # Initialize the paramters
  prior = 0.;
  ini_scale = 0.;
  batch_size = 0.;
  MaxItr = 0.;
  test_step = 0.;
  check_step = 0.;
  lr = 0.;
  lambda = 0.;
  lambda_Theta = 0.;
  lambda_Beta = 0.;
  lambda_B = 0.;

  if data_name == "Lastfm1K"
    # Last.fm1K
    if model == "PRPF" || model == "PF" || model == "pointPRPF"
      prior = (0.3, 0.3, 0.3, 0.3, 0.3, 0.3);
      ini_scale = prior[1]/100;
      batch_size = 100;
      MaxItr = 60;
      test_step = 5;
      check_step = 5;
    elseif model == "LogMF"
      prior = (0.3, 0.3);
      ini_scale = prior[1]/100;
      batch_size = 100;
      lr = 0.000001;
      lambda = 0;
      alpha = 1;
      test_step = 800;
      MaxItr = 5000;
      check_step = 400;
    elseif model == "ListPMF"
      lambda   = 0.001;
      lambda_Theta = 1;
      lambda_Beta = 1;
      lambda_B = 1;
      test_step = 5;
      MaxItr = 3000;
      check_step = 10;
    elseif model == "BPR"
      lr = 0.2;       # learning rate
      lambda = 0;     # regularization weight
      prior = (0.3, 0.3);
      ini_scale = prior[1]/100;
      Itr_step = 2000;
      MaxItr = 2500 * Itr_step;
    end

  elseif data_name == "Lastfm2K"
    # Last.fm2K
    if model == "PRPF" || model == "PF" || model == "pointPRPF"
      prior = (0.3, 0.3, 0.3, 0.3, 0.3, 0.3);
      ini_scale = prior[1]/100;
      batch_size = 100;
      MaxItr = 60;
      test_step = 5;
      check_step = 5;
    elseif model == "LogMF"
      prior = (0.3, 0.3);
      ini_scale = prior[1]/100;
      batch_size = 100;
      lr = 0.000001;
      lambda = 0;
      alpha = 1;
      test_step = 800;
      MaxItr = 5000;
      check_step = 400;
    elseif model == "ListPMF"
      lambda   = 0.001;
      lambda_Theta = 1;
      lambda_Beta = 1;
      lambda_B = 1;
      test_step = 5;
      MaxItr = 3000;
      check_step = 10;
    elseif model == "BPR"
      lr = 0.2;       # learning rate
      lambda = 0;     # regularization weight
      prior = (0.3, 0.3);
      ini_scale = prior[1]/100;
      Itr_step = 2000;
      MaxItr = 2500 * Itr_step;
    end

  elseif data_name == "Lastfm360K"
    # Last.fm2K
    if model == "PRPF" || model == "PF" || model == "pointPRPF"
      prior = (0.3, 0.3, 0.3, 0.3, 0.3, 0.3);
      ini_scale = prior[1]/100;
      batch_size = 0;
      MaxItr = 60;
      test_step = 5;
      check_step = 5;
    elseif model == "LogMF"
      prior = (0.3, 0.3);
      ini_scale = prior[1]/100;
      batch_size = 100;
      lr = 0.000001;
      lambda = 0;
      alpha = 1;
      test_step = 800;
      MaxItr = 5000;
      check_step = 400;
    elseif model == "ListPMF"
      lambda   = 0.001;
      lambda_Theta = 1;
      lambda_Beta = 1;
      lambda_B = 1;
      test_step = 5;
      MaxItr = 3000;
      check_step = 10;
    elseif model == "BPR"
      lr = 0.2;       # learning rate
      lambda = 0;     # regularization weight
      prior = (0.3, 0.3);
      ini_scale = prior[1]/100;
      Itr_step = 2000;
      MaxItr = 2500 * Itr_step;
    end

  elseif data_name == "MovieLens100K"
    # MovieLens100K
    if model == "PRPF" || model == "PF" || model == "pointPRPF"
      prior = (0.3, 0.3, 0.3, 0.3, 0.3, 0.3);
      ini_scale = prior[1]/100;
      usr_batch_size = 0
      test_step = 2
      check_step = 2
      MaxItr = 40
    elseif model == "LogMF"
      prior = (0.3, 0.3);
      ini_scale = prior[1]/100;
      batch_size = 100;
      lr = 0.001;
      lambda = 0;
      alpha = 1;
      test_step = 50;
      MaxItr = 3000;
      check_step = 50;
    elseif model == "ListPMF"
      lambda   = 0.001;
      lambda_Theta = 0.01;
      lambda_Beta = 0.01;
      lambda_B = 0.01;
      test_step = 5;
      MaxItr = 200;
      check_step = 5;
    elseif model == "BPR"
      lr = 0.2;          # learning rate
      lambda = 0;        # regularization weight
      prior = [0.3 0.3];
      ini_scale = prior[1]/100;
      Itr_step = 2000;
      MaxItr = 2500 * Itr_step;
    end

  elseif data_name == "MovieLens1M"
    # MovieLens1M
    if model == "PRPF" || model == "PF" || model == "pointPRPF"
      prior = (0.3, 0.3, 0.3, 0.3, 0.3, 0.3);
      ini_scale = prior[1]/100;
      usr_batch_size = 0
      test_step = 5
      check_step = 5
      MaxItr = 80
    elseif model == "LogMF"
      prior = (0.3, 0.3);
      ini_scale = prior[1]/100;
      batch_size = 100;
      lr = 0.001;
      lambda = 0;
      alpha = 1;
      test_step = 1000;
      MaxItr = 50000;
      check_step = 500;
    elseif model == "ListPMF"
      lambda   = 0.01;
      lambda_Theta = 0.1;
      lambda_Beta = 0.1;
      lambda_B = 0.1;
      test_step = 5;
      MaxItr = 200;
      check_step = 5;
    elseif model == "BPR"
      lr = 0.1;         # learning rate
      lambda = 0.001;   # regularization weight
      prior = [0.3 0.3];
      ini_scale = prior[1]/100;
      Itr_step = 2000;
      MaxItr = 1000 * Itr_step;
    end

  end

  return (prior, ini_scale, batch_size, MaxItr, test_step, check_step, lr,
          lambda, lambda_Theta, lambda_Beta, lambda_B)
end