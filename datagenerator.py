"""
real data
tile 3x3
segment length : 2s
video length :
number of segments :
qp set : '19','23','27','31','35','43','51'
file size :
bitrate :
"""
import random
import video_info
#import roi_info
random.seed(1)
import viewportdatagenerator as vdg
import numpy as np
np.random.seed(1)
import math
import json
from scipy.stats import lognorm
num_p_samples=100
num_seg_list_indisk = [60,62,32 ,84, 206]
num_tile_per_seg=24

#tot_num_video=200
tot_num_video=500
start_in_seg_per_video=np.zeros(tot_num_video)
num_ver_per_tile=7
num_ver_per_seg=num_tile_per_seg*num_ver_per_tile

#num_segs_every_video=np.random.randint(30,100,tot_num_video)
num_segs_every_video=np.random.randint(30,900,tot_num_video)
num_segs=sum(num_segs_every_video)



print('tot seg : %d'%num_segs)
space_limit=30000
global_theta=0.3
roi_mean=0.7
roi_stdev=0.1


def getSegTilePData():
    video_names = ['Elephants', 'Diving', 'Rhinos', 'Rollercoaster', 'Timelapse', 'Venice', 'Paris']
    forder_path = './txtdata/roidata/roi_data/'
    tile_p_list = []

    for video_idx in range(len(video_names)):
        video_data_path = forder_path + video_names[video_idx] + '.txt'
        seg_cnt = 0

        with open(video_data_path, 'r') as file:

            while True:
                line = file.readline()
                tile_p = []
                if not line:
                    break
                str_line = line.strip().split(' ')
                for str_idx in range(len(str_line)):
                    tile_p.append(float(str_line[str_idx]))
                seg_cnt += 1
                tile_p_list.append(tile_p)
                if (seg_cnt >= 35):
                    break
                # print(line.strip())
    #return tile_p_list[180:],roi_map_list[180:]
    return tile_p_list




def generateData():
    np.random.seed(1)
    random.seed(1)
    tot_num_seg=sum(num_segs_every_video)
    tot_num_ver=tot_num_seg*num_tile_per_seg*num_ver_per_tile
    file_size=np.zeros(tot_num_ver)
    bitrate=np.zeros(tot_num_ver)
    qoe=np.zeros((tot_num_ver))
    #vmaf_mean = [100,95, 88, 75, 60, 45, 29]
    vmaf_mean = [100,93, 89, 85, 78, 63, 29]
    br_list=[2,1.5,1,0.8,0.6,0.4,0.2]

    vmaf_stdev=[0,2,7,7,7,7,7]
    for i in range(tot_num_video):
        selected_video_no=0#video_base_list[i]
        num_seg_generate=num_segs_every_video[i]

        if(i>0):
            start_in_seg_per_video[i]=start_in_seg_per_video[i-1]+num_segs_every_video[i-1]
        for j in range(num_seg_generate):


            video_start_in_ver=int(start_in_seg_per_video[i]*num_ver_per_seg)
            seg_start_in_ver = int(video_start_in_ver+j*num_ver_per_seg)
            seg_end_in_ver = seg_start_in_ver + num_ver_per_seg
            qoe_for_seg=np.zeros(num_ver_per_seg,dtype=float)
            bw_for_seg=np.zeros(num_ver_per_seg,dtype=float)
            file_size_for_seg=np.zeros(num_ver_per_seg,dtype=float)

            for t in range(num_tile_per_seg):
                #print('%d , %d ,%d'%(i,j,t))
                start_tile_idx=t*num_ver_per_tile
                v=0
                qoe_tile_ver = np.zeros(num_ver_per_tile, dtype=float)
                bw_tile_ver = np.zeros(num_ver_per_tile, dtype=float)
                file_size_tile_ver = np.zeros(num_ver_per_tile, dtype=float)

                for v_idx in range(num_ver_per_tile):

                    qoe_mean = video_info.video_qoe_info[selected_video_no][v_idx][0]
                    qoe_diff = video_info.video_qoe_info[selected_video_no][v_idx][3]
                    qoe_max = video_info.video_qoe_info[selected_video_no][v_idx][1]
                    qoe_min = video_info.video_qoe_info[selected_video_no][v_idx][2]

                    bw_mean = video_info.video_bitrate_info[selected_video_no][v_idx][0]
                    bw_diff = video_info.video_bitrate_info[selected_video_no][v_idx][3]
                    bw_max = video_info.video_bitrate_info[selected_video_no][v_idx][1]
                    bw_min = video_info.video_bitrate_info[selected_video_no][v_idx][2]
                    tmp_qoe = np.random.normal(vmaf_mean[v_idx], vmaf_stdev[v_idx])
                    tmp_bw =br_list[v_idx]

                    if v==0:
                        tmp_qoe=100
                    else:
                        while (v > 0 and (tmp_qoe <= tmp_qoe-7 or tmp_qoe > tmp_qoe+7 or tmp_qoe >= qoe_tile_ver[v - 1])):
                            tmp_qoe = np.random.normal(vmaf_mean[v_idx], vmaf_stdev[v_idx])
                        # if(v==6):
                        #     tmp_qoe+=10
                    tmp_fz = 2 * tmp_bw/8
                    qoe_tile_ver[v]=tmp_qoe
                    bw_tile_ver[v]=tmp_bw

                    file_size_tile_ver[v]=tmp_fz
                    v+=1

                qoe_for_seg[start_tile_idx:start_tile_idx+num_ver_per_tile]=qoe_tile_ver
                bw_for_seg[start_tile_idx: start_tile_idx + num_ver_per_tile]=bw_tile_ver
                file_size_for_seg[start_tile_idx:start_tile_idx+num_ver_per_tile]=file_size_tile_ver

            file_size[seg_start_in_ver:seg_end_in_ver]=file_size_for_seg[:num_ver_per_seg]
            bitrate[seg_start_in_ver:seg_end_in_ver]=bw_for_seg[:num_ver_per_seg]
            qoe[seg_start_in_ver:seg_end_in_ver]=qoe_for_seg[:num_ver_per_seg]

    print('tottal video space %.4f'%(sum(file_size)))

    return file_size,bitrate,qoe

def generateSampleRoiP(_seed=5):
    random.seed(_seed)
    np.random.seed(_seed)
    dataset_tile_p=getSegTilePData()
    select_segs_dataset=np.random.randint(0,len(dataset_tile_p),num_segs)
    tile_popularity = []#np.zeros(num_segs)
    for i in range(num_segs):
        tile_popularity.append(np.copy(dataset_tile_p[i]).tolist())
    return tile_popularity

def generateRoiPopularity(_seed=1):
    """
    video seg인기도를 고려하지 않는 viewport 및 비 viewport 타일들의 인기도 데이터를 생성한다.

    :return: roi_tiles_list,roi_popularity,roi_info_idx_list
    """
    random.seed(1)
    np.random.seed(_seed)
    tot_vers=num_segs*num_ver_per_seg
    random.seed(_seed)
    np.random.seed(_seed)
    dataset_tile_p = getSegTilePData()
    select_segs_dataset = np.random.randint(0, len(dataset_tile_p), num_segs)
    tile_popularity = []  # np.zeros(num_segs)
    for i in range(num_segs):
        tile_popularity.append(np.copy(dataset_tile_p[select_segs_dataset[i]]).tolist())
    return tile_popularity

#generateViewPortPopularity()
def zipf_distribution_popularity(VIDEO_THETA,N):
    """
    :param VIDEO_THETA: theta 작을 수록 소수의 item에 인기도를 쏠린다.
    :param N:
    :return:
    """
    popularity=[]
    gFactor = 0
    for i in range(1,N+1):
        gFactor += 1 / math.pow(i, 1 - VIDEO_THETA)
    gFactor = 1.0 / gFactor
    for i in range(N):
        popularity.append(gFactor / math.pow(i + 1, 1 - VIDEO_THETA));
    return popularity
def zipf_distribution_segment(theta,vec_size,partial_size):
    vec=[]
    zipf_size = vec_size // partial_size
    if (vec_size % partial_size):
        zipf_size+=1
    gFactor = 0;

    for i in range(1,zipf_size+1):
        gFactor += 1 /math.pow(i, 1-theta);

    gFactor = 1.0 / gFactor;

    for i in range(zipf_size):
        tmp = gFactor/pow(i + 1, 1-theta);
        ssize = partial_size;
        if ((i + 1) * partial_size > vec_size):
            ssize -= ((i + 1) * partial_size) %vec_size;
        for cnt in range(ssize):
            vec.append(tmp)
    return vec


def getSegsPopularityList(no_rank_list):
    """
    비디오 인기도를 기반하여 비디오마다 세그먼트의 인기도를 도출한다.
    :return: 모든 세그먼트의 인기도 리스트
    """
    theta=global_theta
    video_popularity=zipf_distribution_popularity(theta,tot_num_video)
    #print(video_popularity[:100])
    segs_popularity=np.zeros(num_segs)
    video_start_in_seg=0
    for i in range(len(num_segs_every_video)):
        video_end_in_seg=video_start_in_seg+num_segs_every_video[i]
        seg_popularity_per_video=zipf_distribution_segment(0.3,num_segs_every_video[i],30)
        #seg_popularity_per_video=zipf_distribution_popularity(theta,num_segs_every_video[i])
        for j in range(len(seg_popularity_per_video)):
            seg_popularity_per_video[j] *= (video_popularity[no_rank_list[i]])

        segs_popularity[video_start_in_seg:video_end_in_seg]=seg_popularity_per_video
        video_start_in_seg+=num_segs_every_video[i]

    # num_req_per_seg=[]
    # with open('./txtdata/' + str(int(global_theta * 10)) + 'numrequests.txt', 'r') as f_req:
    #     num_req_per_seg_str = f_req.readline()
    # num_req_per_seg_split = num_req_per_seg_str.split(' ')
    # for i in range(num_segs):
    #     num_req_per_seg.append(int(num_req_per_seg_split[i]))
    # tot_num_req=sum(num_req_per_seg)
    # for i in range(num_segs):
    #     segs_popularity[i]=num_req_per_seg[i]/tot_num_req
    return segs_popularity
def getTilesPopularity(seed=1):
    """

    :return: 타일 인기도
    """
    tile_popularity=generateRoiPopularity(seed)

    return tile_popularity
def getPredictTilePpopularity():
    file_path="./mylstm/predicted_tile_popularity.txt"
    # 1. 读取文件 -> (总seg数, num_tile_per_seg)
    dataset_tile_p = np.loadtxt(file_path, dtype=float)

    # 2. 在总seg中随机选取 num_segs 个
    total_segs = dataset_tile_p.shape[0]
    select_indices = np.random.choice(total_segs, size=num_segs, replace=True)

    # 3. 取出对应行
    tile_popularity = [dataset_tile_p[idx].tolist() for idx in select_indices]

    return tile_popularity
    pass
def getTileP_list(_tile_popularity):

    vp_tiles_list = []
    vp_bitmap = []
    tile_popularity=[]
    for i in range(num_segs):
        tile_popularity.append(np.zeros(num_tile_per_seg,dtype=float).tolist())
    # 세그먼트마다 미리 viewport생성
    for i in range(num_segs):
        vp_tiles = vdg.viewportDataGenerator(num_tile_per_seg, _tile_popularity[i],
                                             20)
        bitmap = []

        for r in range(20):
            bitmap_per_request = []
            for j in range(num_tile_per_seg):
                if j in vp_tiles[r]:
                    tile_popularity[i][j]+=1
                    bitmap_per_request.append(1)
                else:
                    bitmap_per_request.append(0)
            bitmap.append(bitmap_per_request)
        vp_bitmap.append(bitmap)
        sum_req=sum(tile_popularity[i])
        for tile in range(num_tile_per_seg):
            tile_popularity[i][tile]=tile_popularity[i][tile]/sum_req
        vp_tiles_list.append(vp_tiles)
    return tile_popularity
def getBandwidthDistribution():
    np.random.seed(1)
    #seg_popularity=getSegsPopularityList()
    num_request=50000
    bw_distribution=[]
    request_cnt=0
    bw_mean=13
    bw_stdev=2
    cnt1=0
    num_req_per_seg=[]
    num_req_per_seg_str=''
    with open('./txtdata/dynamic/' + str(int(global_theta * 10)) + 'numrequests.txt', 'r') as f_req:
            num_req_per_seg_str=f_req.readline()
    num_req_per_seg_split=num_req_per_seg_str.split(' ')
    for i in range(num_segs):
        num_req_per_seg.append(int(num_req_per_seg_split[i]))

    for i in range(num_segs):
        num_request_per_seg=num_req_per_seg[i]#int(num_request*seg_popularity[i])
        if(num_request_per_seg)<1 :
            num_request_per_seg=1
            bw_distribution.append([])
            continue
        request_cnt+=int(num_request_per_seg)
        #np.random.uniform(10,20,num_request_per_seg)#
        if(num_request_per_seg==1):
            cnt1+=1
        sub_bw_distribution=np.zeros(num_request_per_seg)
        for j in range(num_request_per_seg):

            tmp_bw=np.random.normal(bw_mean,bw_stdev)#np.random.normal(30,20,num_request_per_seg)
            while(tmp_bw<=5 or tmp_bw > 15):
                tmp_bw=np.random.normal(bw_mean,bw_stdev)
            sub_bw_distribution[j]=(tmp_bw)

        bw_distribution.append(sub_bw_distribution)

    print('num request for per seg==1 cnt : %d'%(cnt1))
    print(request_cnt)
    return bw_distribution
def getTrainBandwidthClass(_num_request,bw_mean=13,bw_stdev=2):
    np.random.seed(1)
    num_request = _num_request
    bw_class = []
    request_cnt = 0

    for i in range(int(num_request)):
        bw = np.random.normal(bw_mean, bw_stdev)  # np.random.normal(30,20,num_request_per_seg)
        #while (bw <= bw_mean-bw_stdev or bw > bw_mean+bw_stdev):
        while (bw <= bw_mean-bw_stdev or bw > bw_mean+bw_stdev):
            bw = np.random.normal(bw_mean, bw_stdev)
        bw_class.append(bw)

    return bw_class
def sample_logn_trunc(n, mu_log, sigma_log, tau, seed=None):
    """从左截断对数正态(>=tau)生成 n 个样本（单位：MB）"""
    rng = np.random.default_rng(seed)
    Ftau = lognorm.cdf(tau, s=sigma_log, loc=0, scale=np.exp(mu_log))
    u = rng.uniform(Ftau, 1.0, size=n)  # 截断后的概率积分变换
    return lognorm.ppf(u, s=sigma_log, loc=0, scale=np.exp(mu_log))
def getPredictRankData():
    zipf_prob = zipf_distribution_popularity(global_theta, tot_num_video)

    # 读取排名文件
    rank_file = "./mylstm/video_rank_per_timestep_real4.json"
    with open(rank_file, "r", encoding="utf-8") as f:
        video_ranks = json.load(f)  # {vid: [rank0, rank1, ...]}

    T = len(next(iter(video_ranks.values())))  # 时间段数
    N = len(video_ranks)                       # 视频数

    # 在函数里自己创建 no_rank_samples_list
    no_rank_samples_list = [[0]*N for _ in range(T)]

    for vid_str, ranks in video_ranks.items():
        vid = int(vid_str)
        for t, rank in enumerate(ranks):
            no_rank_samples_list[t][vid] = rank

    # 生成 trainPopularityset
    trainPopularityset = []
    for row in no_rank_samples_list:
        #print('row len',len(row))
        seg_popularity = getSegsPopularityList(row)
        trainPopularityset.append(seg_popularity)

    return trainPopularityset
    pass
import csv

def getRealDatasetRankData(rank_csv_path="./realdata/videorank4.csv",forlstm=False):
    """
    读取“每行列位置=编号、值=rank(0..K-1)”的CSV。
    CSV 形如：Hour, 0, 1, 2, ..., K-1
      - 第1列为小时字符串
      - 第i+1列为“编号 i”在该小时的排名（0=第一名）
    返回：trainPopularityset = [ getSegsPopularityList(ranks_at_t) for each hour t ]
    """
    no_rank_samples_list = []  # list[list[int]]，t 行，对应每小时的 [rank_by_id0, rank_by_id1, ...]

    with open(rank_csv_path, "r", encoding="utf-8") as f:
        rdr = csv.reader(f)
        header = next(rdr, None)
        if not header:
            raise ValueError("CSV 为空")

        K = len(header) - 1  # 除去 Hour 列后的列数 = 编号总数

        for line_idx, row in enumerate(rdr, start=2):  # 从第2行数据开始（含表头为第1行）
            if not row:
                continue
            ranks = row[1:]
            if len(ranks) != K:
                raise ValueError(f"第 {line_idx} 行列数不一致：{len(ranks)} != {K}")

            # 兼容可能的字符串/浮点文本
            try:
                ranks = [int(float(x)) for x in ranks]
            except Exception as e:
                raise ValueError(f"第 {line_idx} 行存在非数值：{e}")

            # 自动识别是否是 1..K 的排名并矫正为 0..K-1
            rmin, rmax = min(ranks), max(ranks)
            if rmin == 1 and rmax == K:
                ranks = [r - 1 for r in ranks]
            elif rmin != 0 or rmax != K - 1:
                # 如果既不是0..K-1也不是1..K，给出警告（可以改成raise）
                print(f"[WARN] 第 {line_idx} 行 ranks 范围异常：min={rmin}, max={rmax} (期望 0..{K-1})")

            # 可选：校验是否是一个排列（每个名次只出现一次）
            # if sorted(ranks) != list(range(K)):
            #     print(f"[WARN] 第 {line_idx} 行 ranks 非排列：{ranks[:10]} ...")

            no_rank_samples_list.append(ranks)

    # 生成 trainPopularityset（与你原逻辑一致）
    trainPopularityset = []
    for ranks_at_t in no_rank_samples_list:
        seg_popularity = getSegsPopularityList(ranks_at_t)
        trainPopularityset.append(seg_popularity)
    if(forlstm==True):
        return trainPopularityset,no_rank_samples_list
    return trainPopularityset

"""
normal distribution에 따라 인기도 변한 후에 랭킹
mu 항상 0으로 설정
@return rank_no list
"""
def change_video_popularity_rank(rank_no_list,zipf_prob,mu,sigma,seed_flag=0,rng=None):
    #np.random.seed(1)
    #random.seed(1)
    video_no_prob_list=[]
    num_video=len(rank_no_list)
    for i in range(num_video):
        val=np.random.normal(mu,sigma)
        #list원소 : (video번호,video변한 뒤의 인기도)
        video_no_prob_list.append((rank_no_list[i],zipf_prob[i]+val))
    video_no_prob_list.sort(key=lambda video_no_prob_list: video_no_prob_list[1],reverse=True)
    video_no_prob_list=np.array(video_no_prob_list,dtype=int)
    rank_no_list=video_no_prob_list[:,0]
    return rank_no_list
def change_video_popularity_rank_flat(rank_no_list,zipf_prob,distribution,group_size,seed_flag=0,rng=None):
    np.random.seed(1)
    random.seed(1)
    video_no_prob_list=[]
    num_video=len(rank_no_list)

    for i in range(num_video):
        group_no=i//group_size
        mu, sigma=distribution[group_no]
        val=np.random.normal(mu,sigma)
        #list원소 : (video번호,video변한 뒤의 인기도)
        video_no_prob_list.append((rank_no_list[i],zipf_prob[i]+val))
    video_no_prob_list.sort(key=lambda video_no_prob_list: video_no_prob_list[1],reverse=True)
    video_no_prob_list=np.array(video_no_prob_list,dtype=int)
    rank_no_list=video_no_prob_list[:,0]
    return rank_no_list
"""
훈련을 빠르게 하기 위해 변경된 video rank순위 데이터를 미리 생성한다.
@param: rank_no_list : 변경된 video 순위 정보를 저장하는 리스트 
@param: zipf_prob : 순위에 따른 zipf인기도 정보 리스트
@param: 인기도를 변경할 때 필요한 normal분포의 인자나 인자들 (mu,sigma): 
    split_unit==0일 때 랭킹를 변경할 때 모든 비디오를 대상으로 랭킹을 변경한다.
    split_unit!=0일 때 랭킹를 변경할 때 비디오들을 몇 부분으로 짤라, 각 부분 안에만 랭킹이 변경된다. 아직 구현하지 않음
@param: 비디오 랭킹을 변할 패턴 결정 변수, default 값을 0으로 된다.
@return 변환 랭킹에 따른 비디오 번호 및 랭킹정보
"""
def get_video_rank_info_samples(rank_no_list, zipf_prob, normal_factors, num_samples, split_unit,isShuffle=1):
    no_rank_samples_list=[]
    rng = np.random.default_rng()
    if(isShuffle==0):
        #전체 비디오를 대상으로 ranking변화를 줄 때
        for i in range(num_samples):
            temp_rank_no_list = []
            temp_rank_no_list = change_video_popularity_rank_flat(rank_no_list, zipf_prob,normal_factors,split_unit,rng).tolist()
            temp_no_rank_list = np.zeros(len(zipf_prob)).tolist()
            for rank in range(len(zipf_prob)):
                video_no = temp_rank_no_list[rank]
                temp_no_rank_list[video_no] = rank
            no_rank_samples_list.append(temp_no_rank_list)
    else:
        for i in range(num_samples):
            shuffle_video_no = 0
            tmp_no_rank_samples_list=np.arange(tot_num_video).tolist()
            shuffle_group_idx=0
            while (shuffle_video_no < tot_num_video):
                shuffle_end = shuffle_video_no + split_unit
                if (shuffle_end > tot_num_video):
                    shuffle_end = tot_num_video
                # print(shuffle_video_no)
                # print(shuffle_end)
                normal_factor=normal_factors[shuffle_group_idx]
                tmp_no_rank_list = change_video_popularity_rank(rank_no_list[shuffle_video_no:shuffle_end],
                                                                       zipf_prob[shuffle_video_no:shuffle_end],
                                                                       normal_factor[0], normal_factor[1],rng)
                tmp_no_rank_samples_list[shuffle_video_no:shuffle_end] = tmp_no_rank_list
                shuffle_video_no += split_unit
                shuffle_group_idx+=1
            no_rank_samples_list.append(tmp_no_rank_samples_list)
    return no_rank_samples_list

def generate_video_rank_train_data(_num_p_samples=30,forlstm=False):
    rank_no_list=np.arange(tot_num_video).tolist()
    zipf_prob=zipf_distribution_popularity(global_theta,tot_num_video)

    split_unit=30
    normal_factors=[]
    num_group=tot_num_video//split_unit
    if(num_group<tot_num_video/split_unit):
        num_group+=1
    sigma_start=0.25
    for group in range(num_group):
        sigma=sigma_start-0.02*group
        if (sigma < 0.05):
            sigma = 0.05
        normal_factor=(0,sigma)

        normal_factors.append(normal_factor)
    #normal_factors=[(0,0.019),(0,0.017),(0,0.015),(0,0.013)]
    no_rank_samples_list=get_video_rank_info_samples(rank_no_list,zipf_prob,normal_factors,_num_p_samples,split_unit)

    # for i in range(num_p_samples//20):
    #     print(no_rank_samples_list[i][90:100])
    trainPopularityset=[]
    for i in range(len(no_rank_samples_list)):
        seg_popularity=getSegsPopularityList(no_rank_samples_list[i])
        trainPopularityset.append(seg_popularity)
    #print(trainPopularityset[0][:50])
    #print(len(trainPopularityset))
    if(forlstm==True):
        return trainPopularityset,no_rank_samples_list
    return trainPopularityset
    pass
#generate_video_rank_train_data()
# #getTilesPopularity()
#getBandwidthDistribution()

