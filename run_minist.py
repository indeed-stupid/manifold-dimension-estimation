# image dataset
from mymodules import *
from scipy.io import loadmat

global_start_time = time.time()
alphas = [1.01, 1.2, 1.4, 1.6, 1.8, 2, 4, 6, 8, 10]
Ks = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
Ks_DanCo = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
ps = [784]
if True:
    hp_01 = np.zeros((18, 10))
    hp_02 = np.zeros((18, 10))
    hp_03 = np.zeros((18, 10))
    hp_04 = np.zeros((18, 10))
    hp_05 = np.zeros((18, 10))
    hp_07 = np.zeros((18, 10))
    hp_08 = np.zeros((18, 10))
    hp_09 = np.zeros((18, 10)) 
    hp_06 = np.zeros((18, 10))
    hp_10 = np.zeros((18, 10))
    
    hp_11 = np.zeros((18, 10))
    hp_12 = np.zeros((18, 10))
    hp_13 = np.zeros((18, 10))
    hp_14 = np.zeros((18, 10))
    hp_15 = np.zeros((18, 10))
    num = 0
    for _type in [0]:
        if _type == 0:
            data = np.genfromtxt('mnist_train.csv', delimiter=',', skip_header=1)
            # First column is label
            labels = data[:, 0]
            images = data[:, 1:]
            # Select only rows where label == 1
            sample = images[labels == 1]
            n = int(sample.shape[0])
        for count in range(10):
            try:
                lPCA = skdim.id.lPCA(ver='FO').fit_transform_pw(sample, n_neighbors = Ks[count], n_jobs=-1)
                hp_01[_type, count] = np.mean(lPCA)
            except Exception as e:
                print(e)
                while True:
                    try:
                        data = sample + np.random.normal(0, 1e-12, size=sample.shape)
                        lPCA = skdim.id.lPCA(ver='FO').fit_transform_pw(data, n_neighbors = Ks[count], n_jobs=-1)
                        hp_01[_type, count] = np.mean(lPCA)
                        print("correction success")
                        break
                    except Exception as e:
                        print("correction failure")
                        continue
            
            MLE = skdim.id.MLE().fit_transform_pw(sample, n_neighbors = Ks[count], n_jobs=-1)
            hp_02[_type, count] = np.mean(MLE)
            
            # try:
                # DanCo = skdim.id.DANCo(k=Ks_DanCo[count]).fit(sample)
                # hp_03[_type, count] = DanCo.dimension_
            # except Exception as e:
                # print(e)
                # while True:
                    # try:
                        # data = sample + np.random.normal(0, 1e-12, size=sample.shape)
                        # DanCo = skdim.id.DANCo(k=Ks_DanCo[count]).fit(data)
                        # hp_03[_type, count] = DanCo.dimension_
                        # print("correction success")
                        # break
                    # except Exception as e:
                        # print("correction failure")
                        # continue
        
            MADA = skdim.id.MADA().fit_transform_pw(sample, n_neighbors=Ks[count], n_jobs=-1)
            hp_04[_type, count] = np.mean(MADA)
            TLE = skdim.id.TLE().fit_transform_pw(sample, n_neighbors=Ks[count], n_jobs=-1)
            hp_05[_type, count] = np.mean(TLE)
            hp_07[_type, count] = ESS(-1, ps[_type], n, Ks[count], sample)
            hp_08[_type, count] = ABID(-1, ps[_type], n, Ks[count], sample)
            hp_09[_type, count] = Wasserstein_new(-1, ps[_type], n, -1, sample, alphas[count])         
        
            try:
                hp_13[_type, count] = q_estimator_parallel_v13(ps[_type], n, Ks[count], sample, num_neighborhoods=n)
            except Exception as e:
                print(e)
                while True:
                    try:
                        data = sample + np.random.normal(0, 1e-12, size=sample.shape)
                        hp_13[_type, count] = q_estimator_parallel_v13(ps[_type], n, Ks[count], data, num_neighborhoods=n)
                        print("correction success")
                        break
                    except Exception as e:
                        print("correction failure")
                        continue
        
        
            try:
                hp_14[_type, count] = tls_estimator_parallel_v13(ps[_type], n, Ks[count], sample, num_neighborhoods=n)
            except Exception as e:
                print(e)
                while True:
                    try:
                        data = sample + np.random.normal(0, 1e-12, size=sample.shape)
                        hp_14[_type, count] = tls_estimator_parallel_v13(ps[_type], n, Ks[count], data, num_neighborhoods=n)
                        print("correction success")
                        break
                    except Exception as e:
                        print("correction failure")
                        continue
                        
            try:
                hp_15[_type, count] = CAPCA(ps[_type], n, Ks[count], sample)
            except Exception as e:
                print(e)
                while True:
                    try:
                        data = sample + np.random.normal(0, 1e-12, size=sample.shape)
                        hp_15[_type, count] = CAPCA(ps[_type], n, Ks[count], data)
                        print("correction success")
                        break
                    except Exception as e:
                        print("correction failure")
                        continue


    hps = {
        0:hp_01,
        1:hp_02,
        2:hp_03,
        3:hp_04,
        4:hp_05,
        5:hp_06,
        6:hp_07,
        7:hp_08,
        8:hp_09,
        9:hp_10,
        10:hp_11,
        11:hp_12,
        12:hp_13,
        13:hp_14,
        14:hp_15
    }
    
    ds_01 = np.zeros((18, 100))
    ds_02 = np.zeros((18, 100))
    ds_03 = np.zeros((18, 100))
    ds_04 = np.zeros((18, 100))
    ds_05 = np.zeros((18, 100))
    ds_06 = np.zeros((18, 100))
    ds_07 = np.zeros((18, 100))
    ds_08 = np.zeros((18, 100))
    ds_09 = np.zeros((18, 100))
    ds_10 = np.zeros((18, 100))
    ds_11 = np.zeros((18, 100))
    ds_12 = np.zeros((18, 100))
    ds_13 = np.zeros((18, 100))
    ds_14 = np.zeros((18, 100))
    ds_15 = np.zeros((18, 100))

    ranges = np.zeros((15, 36)) # one row per estimator, two bounds for every manifold, now the other way around!
    
    for index_01 in [0]:
        for index_02 in range(15):
            window = 3  # size of the sliding window
            sd_min = np.inf
            k_min = 0
            k_max = 10
            sd_max = 0
            for i in range(10 - window + 1):
                sd = np.std(hps[index_02][index_01, i:i+window])
                if sd < sd_min:
                    sd_min = sd
                    k_min = i
                    k_max = i + window
                if sd > sd_max:
                    sd_max = sd
            if sd_min > 0 and sd_max / sd_min  < 1.25:
                k_min = 0
                k_max = 10
            ranges[index_02, 2 * index_01] = k_min
            ranges[index_02, 2 * index_01 + 1] = k_max

    print(ranges)
    
    ds = {
        0:ds_01,
        1:ds_02,
        2:ds_03,
        3:ds_04,
        4:ds_05,
        5:ds_06,
        6:ds_07,
        7:ds_08,
        8:ds_09,
        9:ds_10,
        10:ds_11,
        11:ds_12,
        12:ds_13,
        13:ds_14,
        14:ds_15
    }

    num = 0
    
    for _type in [0]:
    
        for index in [0]:
            if _type == 0:
                data = np.genfromtxt('mnist_train.csv', delimiter=',', skip_header=1)
                # First column is label
                labels = data[:, 0]
                images = data[:, 1:]
                # Select only rows where label == 1
                sample = images[labels == 1]
                n = int(sample.shape[0])
            
            for estimator in [0, 1, 3, 4, 5, 6, 8, 12, 13, 14]:
                dim = []
                
                if estimator == 2:
                    parameters = Ks_DanCo[int(ranges[estimator, _type * 2]):int(ranges[estimator, _type * 2 + 1])]
                if estimator == 8:
                    parameters = alphas[int(ranges[estimator, _type * 2]):int(ranges[estimator, _type * 2 + 1])] # need to check
                if estimator in [5, 9]:
                    parameters = [0]
                if estimator not in [2, 5, 8, 9]:
                    parameters = Ks[int(ranges[estimator, _type * 2]):int(ranges[estimator, _type * 2 + 1])]
                
                for K in parameters:
                    if estimator == 0:
                        try:
                            lPCA = skdim.id.lPCA(ver='FO').fit_transform_pw(sample, n_neighbors = K, n_jobs=-1)
                            dim.append(np.mean(lPCA))
                        except Exception as e:
                            print(e)
                            while True:
                                try:
                                    data = sample + np.random.normal(0, 1e-12, size=sample.shape)
                                    lPCA = skdim.id.lPCA(ver='FO').fit_transform_pw(data, n_neighbors = K, n_jobs=-1)
                                    dim.append(np.mean(lPCA))
                                    print("correction success")
                                    break
                                except Exception as e:
                                    print("correction failure")
                                    continue
                        
                    if estimator == 1:
                        MLE = skdim.id.MLE().fit_transform_pw(sample, n_neighbors = K, n_jobs=-1)
                        dim.append(np.mean(MLE))
                    # if estimator == 2:
                        # try:
                            # DanCo = skdim.id.DANCo(k=K).fit(sample)
                            # dim.append(DanCo.dimension_)
                        # except Exception as e:
                            # print(e)
                            # while True:
                                # try:
                                    # data = sample + np.random.normal(0, 1e-12, size=sample.shape)
                                    # DanCo = skdim.id.DANCo(k=K).fit(data)
                                    # dim.append(DanCo.dimension_)
                                    # print("correction success")
                                    # break
                                # except Exception as e:
                                    # print("correction failure")
                                    # continue
                    if estimator == 3:
                        MADA = skdim.id.MADA().fit_transform_pw(sample, n_neighbors=K, n_jobs=-1)
                        dim.append(np.mean(MADA))
                    if estimator == 4:
                        TLE = skdim.id.TLE().fit_transform_pw(sample, n_neighbors=K, n_jobs=-1)
                        dim.append(np.mean(TLE))
                    if estimator == 6:
                        dim.append(np.mean(ESS(-1, ps[_type], n, K, sample)))
                    if estimator == 7:
                        dim.append(np.mean(ABID(-1, ps[_type], n, K, sample)))
                    if estimator == 8:
                        dim.append(np.mean(Wasserstein_new(-1, ps[_type], n, -1, sample, K)))
                    
                    if estimator == 12:
                        try:
                            dim.append(q_estimator_parallel_v13(ps[_type], n, K, sample, num_neighborhoods=n))
                        except Exception as e:
                            print(e)
                            while True:
                                try:
                                    data = sample + np.random.normal(0, 1e-12, size=sample.shape)
                                    dim.append(q_estimator_parallel_v13(ps[_type], n, K, data, num_neighborhoods=n))
                                    print("correction success")
                                    break
                                except Exception as e:
                                    print("correction failure")
                                    continue
                    
                    if estimator == 13:
                        try:
                            dim.append(tls_estimator_parallel_v13(ps[_type], n, K, sample, num_neighborhoods=n))
                        except Exception as e:
                            print(e)
                            while True:
                                try:
                                    data = sample + np.random.normal(0, 1e-12, size=sample.shape)
                                    dim.append(tls_estimator_parallel_v13(ps[_type], n, K, data, num_neighborhoods=n))
                                    print("correction success")
                                    break
                                except Exception as e:
                                    print("correction failure")
                                    continue
                                    
                    if estimator == 14:
                        try:
                            dim.append(CAPCA(ps[_type], n, K, sample))
                        except Exception as e:
                            print(e)
                            while True:
                                try:
                                    data = sample + np.random.normal(0, 1e-12, size=sample.shape)
                                    dim.append(CAPCA(ps[_type], n, K, data))
                                    print("correction success")
                                    break
                                except Exception as e:
                                    print("correction failure")
                                    continue
                
                if estimator == 5:
                    dim.append(skdim.id.TwoNN().fit(sample).dimension_)
                
                ds[estimator][_type, index] = np.mean(dim)
                print(n, estimator, np.mean(dim))


    np.savetxt("mnist_01.csv", ds_01, delimiter=",")
    np.savetxt("mnist_02.csv", ds_02, delimiter=",")
    np.savetxt("mnist_03.csv", ds_03, delimiter=",")
    np.savetxt("mnist_04.csv", ds_04, delimiter=",")
    np.savetxt("mnist_05.csv", ds_05, delimiter=",")
    np.savetxt("mnist_06.csv", ds_06, delimiter=",")
    np.savetxt("mnist_07.csv", ds_07, delimiter=",")
    np.savetxt("mnist_08.csv", ds_08, delimiter=",")
    np.savetxt("mnist_09.csv", ds_09, delimiter=",")
    np.savetxt("mnist_10.csv", ds_10, delimiter=",")
    np.savetxt("mnist_11.csv", ds_11, delimiter=",")
    np.savetxt("mnist_12.csv", ds_12, delimiter=",")
    np.savetxt("mnist_13.csv", ds_13, delimiter=",")
    np.savetxt("mnist_14.csv", ds_14, delimiter=",")
    np.savetxt("mnist_15.csv", ds_15, delimiter=",")

global_end_time = time.time()
print(global_end_time - global_start_time)

