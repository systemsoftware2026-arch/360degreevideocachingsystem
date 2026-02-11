import math
import random
import datagenerator as kdg
import numpy as np
import server_degradation
import gc

random.seed(1)
np.random.seed(1)

# 모든 비디오의 총 세그 수

theta = kdg.global_theta
# num_period=24
# segPopularityset = kdg.generate_video_rank_train_data(num_period)
group = 30
videoPopularityset = []


def get_video_p(segPopularityset):
    video_start_in_seg = 0
    num_segs_every_video = kdg.num_segs_every_video
    video_p_list = []
    num_video = kdg.tot_num_video
    for video in range(num_video):
        video_end_in_seg = video_start_in_seg + num_segs_every_video[video]
        video_p = sum(segPopularityset[video_start_in_seg:video_end_in_seg]) / group
        for seg in range(video_start_in_seg, video_end_in_seg):
            segPopularityset[seg]  # /=video_p

        video_start_in_seg = video_end_in_seg
        video_p_list.append(video_p)

    return video_p_list
    pass


# RAND_MAX=1000

random.seed(1)
np.random.seed(1)
# def nextPisson():
#     lamb=0.5
#     l=math.exp((-lamb))
#     p=1.0
#     k=0
#     random.random()
#     while(1):
#         k+=1
#         p*=random.random()
#         if(p<=l):
#             break
#     return k-1
#
# def generateArrival():
#     arrival_time=[]
#     arrival_time.append(nextPisson())
#     i=1
#     while(1):
#         t=arrival_time[i - 1] + nextPisson()
#         if(t>=86400):
#             break
#         arrival_time.append(t)
#         i+=1
#     #print(arrival_time)
#     print(len(arrival_time))
#     return arrival_time
import random
import math

import random
import math


def nextExponential(lamb):
    """
    使用指数分布生成间隔时间
    :param lamb: 到达率 λ
    :return: 指数分布随机变量
    """
    return -math.log(1 - random.random()) / lamb


def generateArrival():
    lamb = 2
    """
	生成伯森过程的到达时间
    :param lamb: 到达率 λ
    :param total_time: 总时间长度
    :return: 到达时间列表
    """
    arrival_time = []
    current_time = 0
    total_time = 86400


    while current_time < total_time:
        interval = nextExponential(lamb)  # 生成间隔时间
        current_time += interval
        #print(int(current_time))
        if current_time < total_time:
            arrival_time.append(int(current_time))

    return arrival_time


def video_choose(video_popularity):
    RAND_MAX = 10000
    rnd = random.randint(1, RAND_MAX)
    for idx in range(0, kdg.tot_num_video):
        rnd -= video_popularity[idx] * RAND_MAX
        if (rnd <= 0):
            return idx
    return 0


def segment_choose(video_no, seg_popularity):
    seg_idx_start = 0
    video_no = (video_no)
    RAND_MAX = 1000  # num_segs_every_video[video_no]
    # print(video_no)
    num_segs_every_video = kdg.num_segs_every_video
    for i in range((video_no)):
        seg_idx_start += num_segs_every_video[i]
    seg_idx_end = seg_idx_start + num_segs_every_video[video_no]
    rnd = random.randint(1, int(RAND_MAX))
    for idx in range(seg_idx_start, seg_idx_end, 30):
        rnd -= seg_popularity[idx] * RAND_MAX
        if (rnd <= 0):
            return idx
    return seg_idx_end


def hours24simulation():
    arrive_time = []
    num_period = 24
    segPopularityset = kdg.getRealDatasetRankData()#generate_video_rank_train_data(num_period)
    video_popularity = []
    period_sample_list = np.random.randint(0, len(segPopularityset), num_period).tolist()
    for period in range(num_period):
        tmp_video_p_list = get_video_p(segPopularityset[period_sample_list[period]])
        total_popularity = sum(tmp_video_p_list)
        tmp_video_p_list = [p / total_popularity for p in tmp_video_p_list]
        video_popularity.append(tmp_video_p_list)

    num_segs_every_video = kdg.num_segs_every_video
    tot_num_segs = kdg.num_segs
    arrive_time = generateArrival()
    #print(arrive_time)
    client = 0
    #exit(1)
    # print(arrive_time)
    num_req_per_seg = []
    num_req_per_video = []
    for seg in range(tot_num_segs):
        num_req_per_seg.append(0)
    for seg in range(kdg.tot_num_video):
        num_req_per_video.append(0)
    video_start_in_seg = np.zeros(kdg.tot_num_video).tolist()
    for video_idx in range(1, kdg.tot_num_video):
        video_start_in_seg[video_idx] = video_start_in_seg[video_idx - 1] + num_segs_every_video[video_idx - 1]

    client_list = []
    remove_client_cnt = 0
    num_service_denial = 0
    tot_num_req = 0
    period_length = 3600
    for sec in range(86400):
        num_req_per_seg_for_sec = np.zeros(tot_num_segs, dtype=int).tolist()
        # print('arrive_time[%d] : %d'%(client,arrive_time[client]))
        while (sec == arrive_time[client]):
            period = sec // period_length
            video = video_choose(video_popularity[period])
            seg = segment_choose(video, segPopularityset[period])
            # print('client : %d video : %d seg : %d'%(client,video,seg))
            # print(sec)

            client_arrive_time = sec
            client_end_time = sec + (seg - video_start_in_seg[video]) * 2 - 1
            if (client_end_time > 86400):
                client_end_time = 86400
            client_request = (client, client_arrive_time, client_end_time, video)
            client_list.append(client_request)

            client += 1

            if (client >= len(arrive_time)):
                break
            seg_idx_start = 0
            for i in range(video):
                seg_idx_start += num_segs_every_video[i]
            for i in range(seg_idx_start, seg):
                num_req_per_seg[i] += 1
            num_req_per_video[video] += 1
        client_idx = 0
        num_request_sec = 0
        requests_for_sec = []
        while (len(client_list) > 0 and client_idx < len(client_list)):
            req = client_list[client_idx]
            client_end_sec = req[2]
            video = req[3]
            client_start_sec = req[1]
            seg = ((sec - client_start_sec) // 2)
            seg += video_start_in_seg[video]
            # 클라이언트가 완료되지 않으면 해당 seg request를 저장한다.
            if (sec <= client_end_sec):
                num_request_sec += 1
                requests_for_sec.append(int(seg))
                num_req_per_seg_for_sec[int(seg)] += 1
                if (sec == client_end_sec):
                    del client_list[client_idx]
                    remove_client_cnt += 1
                    continue
            else:
                remove_client_cnt += 1
                del client_list[client_idx]
                continue
            if (client_idx >= len(client_list)):
                break
            client_idx += 1
        tmp_num_req = len(requests_for_sec)
        print(tmp_num_req)
        if (tmp_num_req > 0):
            file_path='./txtdata/requests_log'+str(int(theta*10))+'_real7.txt'
            write_requests_to_file(requests_for_sec,file_path)

        if (sec % 1000 == 0):
            print('sec : %d, tot_ num_request : %d, remove_client_cnt : %d, num_service_denial : %d' % (
            sec, sum(num_req_per_seg), remove_client_cnt, num_service_denial))
        # print(requests_for_sec)

        remove_client_cnt = 0
        tot_num_req += num_request_sec
        # if(num_request_sec>0):
        # 	num_service_denial+=server_degradation.server_simulation_ver2(seg_placement, kdg.server_bandwidth_list,
        # 									 kdg.server_space_list, num_req_per_seg_for_sec)
        if (client >= len(arrive_time)):
            break
    # with open('./txtdata/dynamic/' + str(int(theta * 10)) + 'numrequests.txt', 'w') as f_req:
    # 	for i in range(len(num_req_per_seg)):
    # 		f_req.write(str(num_req_per_seg[i]))
    # 		f_req.write(' ')
    # print(client)

    print('num req  per video ', num_req_per_video)
    print('num req  per seg ', (num_req_per_seg[:1000]))
    # print('num_service_denial : %d' % (num_service_denial))
    return num_req_per_seg  # ,num_service_denial


# exit(1)
def write_requests_to_file(requests_for_sec, file_path='./txtdata/requests_log.txt'):
    """
    Write the list of requests to a text file in one line, separated by spaces, and add a new line for each write.

    :param requests_for_sec: A list of request data to be written to the file.
    :param file_path: The path to the text file (default: 'requests_log.txt').
    """
    with open(file_path, 'a') as file:
        file.write(' '.join(map(str, requests_for_sec)) + '\n')  # Join list with spaces and add a newline


hours24simulation()
