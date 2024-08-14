import numpy as np
from sklearn.feature_selection import mutual_info_classif
from numpy import linalg as LA

def euclidean_distances(X):
    # Computes the squared Euclidean distance matrix using NumPy.
    norms = np.sum(X ** 2, axis=1)[:, np.newaxis]
    return norms + norms.T - 2 * np.dot(X, X.T)
def construct_kernels(X, kernel_scale_factor=1):
    K_dis = euclidean_distances(np.transpose(X))
    epsilon = kernel_scale_factor * np.median(K_dis[~np.eye(K_dis.shape[0], dtype=bool)])
    K = np.exp(-(K_dis ** 2) / (2 * epsilon ** 2))
    return K


def compute_rw_mi(X, Y, n_walks=100, walk_length=5, kernel_scale_factor=1, jump_prob=0.4, decay_factor=0.4):
    n_samples, n_features_x = X.shape
    n_labels = Y.shape[1]

    if n_walks <= 0:
        raise ValueError("n_walks must be greater than 0")


    A_features = construct_kernels(X, kernel_scale_factor)

    A_labels = construct_kernels(Y, kernel_scale_factor)

    mi_matrix_feature_label = np.zeros((n_features_x, n_labels))
    for i in range(n_features_x):
        for j in range(n_labels):
            mi_matrix_feature_label[i, j] = mutual_info_classif(X[:, i].reshape(-1, 1), Y[:, j])[0]

    mi_min = mi_matrix_feature_label.min()
    mi_max = mi_matrix_feature_label.max()
    mi_normalized = (mi_matrix_feature_label - mi_min) / (mi_max - mi_min)

    P_features = A_features / A_features.sum(axis=1, keepdims=True)
    P_labels = A_labels / A_labels.sum(axis=1, keepdims=True)

    P_features = P_features / np.linalg.norm(P_features, ord=1, axis=1, keepdims=True)
    P_labels = P_labels / np.linalg.norm(P_labels, ord=1, axis=1, keepdims=True)

    P_feature_label = mi_normalized / mi_normalized.sum(axis=1, keepdims=True)

    P_label_to_feature = mi_normalized / mi_normalized.sum(axis=0, keepdims=True)


    Rw = np.zeros((n_features_x, n_labels))

    for _ in range(n_walks):
        start_node = np.random.randint(n_features_x)
        walk = [start_node]

        for _ in range(walk_length):
            curr_node = walk[-1]
            if curr_node < n_features_x:
                if np.random.rand() < jump_prob:
                    next_node = np.random.choice(n_labels, p=P_feature_label[curr_node])
                    next_node += n_features_x
                else:
                    next_node = np.random.choice(n_features_x, p=P_features[curr_node])
            else:
                if np.random.rand() < jump_prob:
                    next_node = np.random.choice(n_features_x, p=P_label_to_feature.T[curr_node - n_features_x])
                else:
                    next_node = np.random.choice(n_labels, p=P_labels[curr_node - n_features_x])
                    next_node += n_features_x
            walk.append(next_node)

        for i in range(len(walk)):
            if walk[i] < n_features_x:
                for j in range(1, len(walk)):
                    if walk[j] >= n_features_x:
                        feature_idx = walk[i]
                        label_idx = walk[j] - n_features_x
                        distance = j - i
                        Rw[feature_idx, label_idx] += decay_factor ** distance * mi_matrix_feature_label[feature_idx, label_idx]

    Rw /= n_walks
    Rw_min = Rw.min()
    Rw_max = Rw.max()
    Rw_normalized = (Rw - Rw_min) / (Rw_max - Rw_min)

    return Rw_normalized
def SCMF(X_train, y_train,Rw, select_nub, alpha, beta, gamma,c,d):
    X_train = np.matrix(X_train)
    y_train = np.matrix(y_train)
    maxIter = 1000
    V_dim = 10
    eps = 2.2204e-16
    n, f = X_train.shape
    n, l = y_train.shape
    k = min(V_dim, l)

    V = abs(np.mat(np.random.rand(n, k)))
    Q = abs(np.mat(np.random.rand(k, f)))
    B = abs(np.mat(np.random.rand(k, l)))

    Q = Q.astype(np.float64)
    B = B.astype(np.float64)
    V = V.astype(np.float64)
    X_train = X_train.astype(np.float64)
    Rw = Rw.astype(np.float64)
    X=X_train
    Y=y_train

    iter = 0
    obj = []
    while iter < maxIter:
        # Update V
        V = np.multiply(V, np.true_divide(alpha * X_train * Q.T+beta*y_train * B.T,
                                            alpha * V * Q * Q.T + beta * V * B * B.T + 1*V +eps))

        # Update Q
        Dtmp = np.sqrt(np.sum(np.multiply(Q.T @ B, Q.T @ B), axis=1) + eps)
        d1 = 0.5 / Dtmp
        D = np.diag(d1.flat)
        Q = np.multiply(Q, np.true_divide(alpha*V.T * X_train+ gamma*B * Rw.T+c*B*Y.T*X, alpha*V.T * V * Q + gamma*B * B.T * Q+ 2*d*B @ B.T @ Q @ D+ c*Q*X.T*X + eps))

        # Update B
        Dtmp = np.sqrt(np.sum(np.multiply(Q.T @ B, Q.T @ B), 1) + eps)
        d1 = 0.5 / Dtmp
        D = np.diag(d1.flat)
        B = np.multiply(B, np.true_divide(beta*V.T * y_train+ gamma*Q * Rw+ c*Q*X.T*Y, beta*V.T * V * B + gamma*Q * Q.T * B+ 2*d*Q @ D @ Q.T @ B+ c*B*Y.T*Y + eps))

        obj.append(alpha *pow(LA.norm(X_train - V * Q, 'fro'),2) + beta * pow(
            LA.norm( y_train -V * B , 'fro'), 2)+ gamma*pow(LA.norm(Rw - Q.T * B, 'fro'), 2)+ 2*d * np.sum(Dtmp,axis=0)+ c * pow(LA.norm(X*Q.T - Y * B.T, 'fro'), 2)
                   +1*pow(LA.norm(V, "fro"), 2))

        if (iter > 1 and (abs(obj[iter] - obj[iter - 1]) < 1e-3 or abs(obj[iter] - obj[iter - 1]) / float(
                abs(obj[iter - 1])) < 1e-3)):
            break
        iter = iter + 1

    return Q.T*B