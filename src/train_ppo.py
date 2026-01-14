import warnings

from train_sac import main


if __name__ == "__main__":
    warnings.warn("train_ppo.py is deprecated; use train_sac.py instead.", stacklevel=2)
    main()
