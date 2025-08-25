from mymodules import *

global_start_time = time.time()
if True:
    ds_01 = np.zeros((10, 100))
    ds_02 = np.zeros((10, 100))
    ds_03 = np.zeros((10, 100))
    ds_04 = np.zeros((10, 100))
    ds_05 = np.zeros((10, 100))
    ds_06 = np.zeros((10, 100))
    ds_07 = np.zeros((10, 100))
    ds_08 = np.zeros((10, 100))
    ds_09 = np.zeros((10, 100))
    ds_10 = np.zeros((10, 100))
    ds_15 = np.zeros((10, 100))
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
        14:ds_15
    }
    num = 0
    d = 5
    p = 10
    R = 1
    n = 1000 # 
    K = int(n / 10)
    for R in [0.001, 0.01, 0.05, 0.1, 1, 5, 10, 20, 50, 100]:
        for index in range(0, 100):
            sampler = SphereSampler(n=n, d=d, p=p, R=R, seed=321, sigma = 0.0)
            sample = sampler.sample_fast()
            for estimator in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 14]:
                dim = []
                
                if estimator == 14:
                    try:
                        dim.append(CAPCA(p, n, K, sample))
                    except Exception as e:
                        print(e)
                        while True:
                            try:
                                data = sample + np.random.normal(0, 1e-12, size=sample.shape)
                                dim.append(CAPCA(p, n, K, data))
                                print("correction success")
                                break
                            except Exception as e:
                                print("correction failure")
                                continue
                
                if estimator == 0:
                    try:
                        lPCA = skdim.id.lPCA(ver='FO').fit_transform_pw(sample, n_neighbors = K, n_jobs=-1)
                        dim.append(np.mean(lPCA))
                    except Exception as e:
                        print(e)
                        while True:
                            try:
                                data = sample + np.random.normal(0, 1e-12, size=sample.shape)
                                lPCA = skdim.id.lPCA(ver='FO').fit_transform_pw(sample, n_neighbors = K, n_jobs=-1)
                                dim.append(np.mean(lPCA))
                                print("correction success")
                                break
                            except Exception as e:
                                print("correction failure")
                                continue
                                
                if estimator == 1:
                    MLE = skdim.id.MLE().fit_transform_pw(sample, n_neighbors = K, n_jobs=-1)
                    dim.append(np.mean(MLE))
                
                if estimator == 2:
                    try:
                        DanCo = skdim.id.DANCo(k=10).fit(sample)
                        dim.append(DanCo.dimension_)
                    except Exception as e:
                        print(e)
                        while True:
                            try:
                                data = sample + np.random.normal(0, 1e-12, size=sample.shape)
                                DanCo = skdim.id.DANCo(k=10).fit(sample, n_jobs=-1)
                                dim.append(DanCo.dimension_)
                                print("correction success")
                                break
                            except Exception as e:
                                print("correction failure")
                                continue
                
                if estimator == 3:
                    MADA = skdim.id.MADA().fit_transform_pw(sample, n_neighbors=K, n_jobs=-1)
                    dim.append(np.mean(MADA))
                
                if estimator == 4:
                    TLE = skdim.id.TLE().fit_transform_pw(sample, n_neighbors=K, n_jobs=-1)
                    dim.append(np.mean(TLE))
                    
                if estimator == 5:
                    dim.append(skdim.id.TwoNN().fit(sample).dimension_)
                
                if estimator == 6:
                    dim.append(np.mean(ESS(-1, p, n, K, sample)))
                    
                if estimator == 7:
                    dim.append(np.mean(ABID(-1, p, n, K, sample)))
                    
                if estimator == 8:
                    dim.append(np.mean(Wasserstein_new(-1, p, n, -1, sample, 5)))
                    
                if estimator == 9:
                    dim.append(np.mean(ISOMAP(-1, p, n, -1, sample)))
                
                ds[estimator][num, index] = np.mean(dim)
                print(n, d, estimator, np.mean(dim))
        num += 1
        
        
        

    np.savetxt("c03_01.csv", ds_01, delimiter=",")
    np.savetxt("c03_02.csv", ds_02, delimiter=",")
    np.savetxt("c03_03.csv", ds_03, delimiter=",")
    np.savetxt("c03_04.csv", ds_04, delimiter=",")
    np.savetxt("c03_05.csv", ds_05, delimiter=",")
    np.savetxt("c03_06.csv", ds_06, delimiter=",")
    np.savetxt("c03_07.csv", ds_07, delimiter=",")
    np.savetxt("c03_08.csv", ds_08, delimiter=",")
    np.savetxt("c03_09.csv", ds_09, delimiter=",")
    np.savetxt("c03_10.csv", ds_10, delimiter=",")
    np.savetxt("c03_15.csv", ds_15, delimiter=",")
global_end_time = time.time()
print(global_end_time - global_start_time)