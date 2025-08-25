# uniform, noise, n = 500
from mymodules import *

global_start_time = time.time()
n = 500
uniform_mark = True
noise_sigma = 0.1
alphas = [1.01, 1.2, 1.4, 1.6, 1.8, 2, 4, 6, 8, 10]
Ks = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
Ks_DanCo = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
ps = [10, 20, 40, 10, 20, 40, 10, 20, 40, 6, 6, 6, 4, 3, 4, 4, 4, 4]
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
    num = 0
    for _type in range(0, 18):
        if _type == 0:
            sampler = SphereSampler(n=n, d=5, p=10, R=1, seed=321, sigma = 0.0)
            sample = sampler.sample_fast()
        if _type == 1:
            sampler = SphereSampler(n=n, d=10, p=20, R=1, seed=321, sigma = 0.0)
            sample = sampler.sample_fast()
        if _type == 2:
            sampler = SphereSampler(n=n, d=20, p=40, R=1, seed=321, sigma = 0.0)
            sample = sampler.sample_fast()
        if _type == 3:
            sampler = BallSampler(n=n, d=5, p=10, R=1, seed=321, sigma = 0.0)
            sample = sampler.sample_fast()
        if _type == 4:
            sampler = BallSampler(n=n, d=10, p=20, R=1, seed=321, sigma = 0.0)
            sample = sampler.sample_fast()
        if _type == 5:
            sampler = BallSampler(n=n, d=20, p=40, R=1, seed=321, sigma = 0.0)
            sample = sampler.sample_fast()
        if _type == 6:
            sampler = NormalSurfaceSampler(n=n, d=5, p=10, sd=0.5, sigma= 0.0, seed=321)
            sample = sampler.sample()
        if _type == 7:
            sampler = NormalSurfaceSampler(n=n, d=10, p=20, sd=0.5, sigma= 0.0, seed=321)
            sample = sampler.sample()
        if _type == 8:
            sampler = NormalSurfaceSampler(n=n, d=20, p=40, sd=0.5, sigma= 0.0, seed=321)
            sample = sampler.sample()
        if _type == 9:
            sampler = CosSinSampler(n=n, d=3, k=0.01, sigma=0.0, seed=321, uniform=True)
            sample = sampler.sample()
        if _type == 10:
            sampler = CosSinSampler(n=n, d=3, k=0.1, sigma=0.0, seed=321, uniform=True)
            sample = sampler.sample()
        if _type == 11:
            sampler = CosSinSampler(n=n, d=3, k=1, sigma=0.0, seed=321, uniform=True)
            sample = sampler.sample()
        if _type == 12:
            sampler = CylinderSampler(n=n, d=2, p=4, seed=321, sigma=0.0)
            sample = sampler.sample(uniform=True)
        if _type == 13:
            sampler = HelixSampler(n=n, p=3, seed=321, sigma=0.0)
            sample = sampler.sample(uniform=True)
        if _type == 14:
            sampler = SwissRollSampler(n=n, p=4, seed=321, sigma=0.0)
            sample = sampler.sample(uniform=True)
        if _type == 15:
            sampler = MobiusBandSampler(n=n, p=4, seed=321, sigma=0.0)
            sample = sampler.sample(uniform=True)
        if _type == 16:
            sampler = TorusSampler(n=n, p=4, seed=321, sigma=0.0)
            sample = sampler.sample(uniform=True)
        if _type == 17:
            sampler = HyperbolicSurfaceSampler(n=n, p=4, seed=321, sigma=0.0)
            sample = sampler.sample(uniform=True)
        
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
            
            try:
                DanCo = skdim.id.DANCo(k=Ks_DanCo[count]).fit(sample)
                hp_03[_type, count] = DanCo.dimension_
            except Exception as e:
                print(e)
                while True:
                    try:
                        data = sample + np.random.normal(0, 1e-12, size=sample.shape)
                        DanCo = skdim.id.DANCo(k=Ks_DanCo[count]).fit(data)
                        hp_03[_type, count] = DanCo.dimension_
                        print("correction success")
                        break
                    except Exception as e:
                        print("correction failure")
                        continue
        
            MADA = skdim.id.MADA().fit_transform_pw(sample, n_neighbors=Ks[count], n_jobs=-1)
            hp_04[_type, count] = np.mean(MADA)
            TLE = skdim.id.TLE().fit_transform_pw(sample, n_neighbors=Ks[count], n_jobs=-1)
            hp_05[_type, count] = np.mean(TLE)
            hp_07[_type, count] = ESS(-1, ps[_type], n, Ks[count], sample)
            hp_08[_type, count] = ABID(-1, ps[_type], n, Ks[count], sample)
            hp_09[_type, count] = Wasserstein_new(-1, ps[_type], n, -1, sample, alphas[count]) 


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
        9:hp_10
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

    ranges = np.zeros((10, 36)) # one row per estimator, two bounds for every manifold, now the other way around!
    
    for index_01 in range(18):
        for index_02 in range(10):
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
        9:ds_10
    }

    num = 0
    
    for _type in range(18):
    
        for index in range(0, 100):
            if _type == 0:
                sampler = SphereSampler(n=n, d=5, p=10, R=1, seed=index, sigma = noise_sigma)
                sample = sampler.sample_fast()
            if _type == 1:
                sampler = SphereSampler(n=n, d=10, p=20, R=1, seed=index, sigma = noise_sigma)
                sample = sampler.sample_fast()
            if _type == 2:
                sampler = SphereSampler(n=n, d=20, p=40, R=1, seed=index, sigma = noise_sigma)
                sample = sampler.sample_fast()
            if _type == 3:
                sampler = BallSampler(n=n, d=5, p=10, R=1, seed=index, sigma = noise_sigma)
                sample = sampler.sample_fast()
            if _type == 4:
                sampler = BallSampler(n=n, d=10, p=20, R=1, seed=index, sigma = noise_sigma)
                sample = sampler.sample_fast()
            if _type == 5:
                sampler = BallSampler(n=n, d=20, p=40, R=1, seed=index, sigma = noise_sigma)
                sample = sampler.sample_fast()
            if _type == 6:
                sampler = NormalSurfaceSampler(n=n, d=5, p=10, sd=0.5, sigma=noise_sigma, seed=index)
                sample = sampler.sample()
            if _type == 7:
                sampler = NormalSurfaceSampler(n=n, d=10, p=20, sd=0.5, sigma=noise_sigma, seed=index)
                sample = sampler.sample()
            if _type == 8:
                sampler = NormalSurfaceSampler(n=n, d=20, p=40, sd=0.5, sigma=noise_sigma, seed=index)
                sample = sampler.sample()
            if _type == 9:
                sampler = CosSinSampler(n=n, d=3, k=0.01, sigma=noise_sigma, seed=index, uniform=uniform_mark)
                sample = sampler.sample()
            if _type == 10:
                sampler = CosSinSampler(n=n, d=3, k=0.1, sigma=noise_sigma, seed=index, uniform=uniform_mark)
                sample = sampler.sample()
            if _type == 11:
                sampler = CosSinSampler(n=n, d=3, k=1, sigma=noise_sigma, seed=index, uniform=uniform_mark)
                sample = sampler.sample()
            if _type == 12:
                sampler = CylinderSampler(n=n, d=2, p=4, seed=index, sigma=noise_sigma)
                sample = sampler.sample(uniform=uniform_mark)
            if _type == 13:
                sampler = HelixSampler(n=n, p=3, seed=index, sigma=noise_sigma)
                sample = sampler.sample(uniform=uniform_mark)
            if _type == 14:
                sampler = SwissRollSampler(n=n, p=4, seed=index, sigma=noise_sigma)
                sample = sampler.sample(uniform=uniform_mark)
            if _type == 15:
                sampler = MobiusBandSampler(n=n, p=4, seed=index, sigma=noise_sigma)
                sample = sampler.sample(uniform=uniform_mark)
            if _type == 16:
                sampler = TorusSampler(n=n, p=4, seed=index, sigma=noise_sigma)
                sample = sampler.sample(uniform=uniform_mark)
            if _type == 17:
                sampler = HyperbolicSurfaceSampler(n=n, p=4, seed=index, sigma=noise_sigma)
                sample = sampler.sample(uniform=uniform_mark)
            
            
            
            for estimator in range(0, 10):
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
                    if estimator == 2:
                        try:
                            DanCo = skdim.id.DANCo(k=K).fit(sample)
                            dim.append(DanCo.dimension_)
                        except Exception as e:
                            print(e)
                            while True:
                                try:
                                    data = sample + np.random.normal(0, 1e-12, size=sample.shape)
                                    DanCo = skdim.id.DANCo(k=K).fit(data)
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
                    if estimator == 6:
                        dim.append(np.mean(ESS(-1, ps[_type], n, K, sample)))
                    if estimator == 7:
                        dim.append(np.mean(ABID(-1, ps[_type], n, K, sample)))
                    if estimator == 8:
                        dim.append(np.mean(Wasserstein_new(-1, ps[_type], n, -1, sample, K))) # K should be alpha here
                
                if estimator == 5:
                    dim.append(skdim.id.TwoNN().fit(sample).dimension_)
                if estimator == 9:
                    dim.append(np.mean(ISOMAP(-1, ps[_type], n, -1, sample)))
                
                ds[estimator][_type, index] = np.mean(dim)
                print(n, estimator, np.mean(dim))


    np.savetxt("comparison_noise_500_01.csv", ds_01, delimiter=",")
    np.savetxt("comparison_noise_500_02.csv", ds_02, delimiter=",")
    np.savetxt("comparison_noise_500_03.csv", ds_03, delimiter=",")
    np.savetxt("comparison_noise_500_04.csv", ds_04, delimiter=",")
    np.savetxt("comparison_noise_500_05.csv", ds_05, delimiter=",")
    np.savetxt("comparison_noise_500_06.csv", ds_06, delimiter=",")
    np.savetxt("comparison_noise_500_07.csv", ds_07, delimiter=",")
    np.savetxt("comparison_noise_500_08.csv", ds_08, delimiter=",")
    np.savetxt("comparison_noise_500_09.csv", ds_09, delimiter=",")
    np.savetxt("comparison_noise_500_10.csv", ds_10, delimiter=",")

global_end_time = time.time()
print(global_end_time - global_start_time)
