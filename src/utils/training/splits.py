"""Create the splits for the training."""

from sklearn.model_selection import train_test_split


def create_splits(ids, cv: int = 1, seed: int = 1999):
    """
    Create the splits for the training.

    Parameters
    ----------
    ids: list
        list of ids to split.
    seed : int
        seed for the random generator.
    cv : int
        number of cross-validation splits.
    """
    if cv > 1:
        x_train, x_val, x_test = [], [], []
        for idx_cv in range(cv):
            x_remaining, x_r_test = train_test_split(
                ids, test_size=0.2, random_state=idx_cv
            )
            x_r_train, x_r_val = train_test_split(
                x_remaining, test_size=0.2, random_state=idx_cv
            )
            x_test.append(x_r_test)
            x_train.append(x_r_train)
            x_val.append(x_r_val)
    else:
        x_remaining, x_test = train_test_split(ids, test_size=0.2, random_state=seed)
        x_train, x_val = train_test_split(x_remaining, test_size=0.2, random_state=seed)
        x_train = [x_train]
        x_val = [x_val]
        x_test = [x_test]
    return x_train, x_val, x_test
