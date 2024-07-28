import numpy as np

def Generate_triplets(X_training, Y1_training, Y2_training, ntripletsperimage=10):
    """
    Generate and shuffle triplets for the training of the model.

    :param X_training: Images containing dendrites and spines
    :param Y1_training:  Labels for the dendrites
    :param Y2_training:  Labels for the spines
    :param ntripletsperimage:  Number of triplets to be done for each anchor image.

    """
    X_triplets = []
    Y1_triplets = []
    Y2_triplets = []
    for i in range(X_training.shape[0]):
        choosen=[] # Remember the negative images that have been selected for current image
        for j in range(ntripletsperimage):
            # Select a random image that should not be selected before and different from the current image
            idx = np.random.randint(0, X_training.shape[0]) # Randomly choose an index for the negative image
            while idx == i or idx in choosen:
                idx = np.random.randint(0, X_training.shape[0]) #Randomly choose an index for the negative image
            # Remember which negative image has been selected
            choosen.append(idx)
            # Add the triplet
            X_triplets.append(X_training[i])
            Y1_triplets.append(Y1_training[i])
            Y2_triplets.append(Y2_training[i])
            X_triplets.append(X_training[i]) # TO DO: Consider augmentation of positive
            Y1_triplets.append(Y1_training[i])
            Y2_triplets.append(Y2_training[i])
            X_triplets.append(X_training[idx])
            Y1_triplets.append(Y1_training[idx])
            Y2_triplets.append(Y2_training[idx])

    # Convert the lists into arrays
    X_triplets=np.array(X_triplets)
    Y1_triplets=np.array(Y1_triplets)
    Y2_triplets=np.array(Y2_triplets)

    ## Shuffle the triplets
    # Step 1: Reshape the array to group triplets
    grouped = X_triplets.reshape(-1, 3, 128, 128,1)
    # Step 2: Generate random permutation
    permutation = np.random.permutation(grouped.shape[0])
    # Step 3: Use the permutation to shuffle the grouped data
    randomized_grouped = grouped[permutation]
    # Step 4: Reshape back to original shape
    X_triplets_randomized = randomized_grouped.reshape(-1, 128, 128, 1)
    # Step 5: Use the same permutation to shuffle the labels
    #   First dendrite labels Y1 by triplets
    grouped_labels = Y1_triplets.reshape(-1, 3, 128, 128,1)
    randomized_grouped_labels = grouped_labels[permutation]
    Y1_randomized = randomized_grouped_labels.reshape(-1, 128, 128, 1)
    #   First spine labels Y2 by triplets
    grouped_labels = Y2_triplets.reshape(-1, 3, 128, 128,1)
    randomized_grouped_labels = grouped_labels[permutation]
    Y2_randomized = randomized_grouped_labels.reshape(-1, 128, 128, 1)

    return X_triplets_randomized, Y1_randomized, Y2_randomized

