import os.path
import random
import numpy as np
import datagenerator as kdg
import service_train_ver2
import service
import viewportdatagenerator as vdg
import time
import roi_info
SEED = 1
random.seed(SEED) #랜덤 시드 고정
np.random.seed(SEED)

#filesize,bitrate,qp
file_size, bitrate, q_list=kdg.generateData()
tot_file_size=sum(file_size)

tile_popularity=kdg.getTilesPopularity()
# print(roi_tiles[5])
# print(roi_tiles[10])
# print(roi_popularity[:10])


num_video=kdg.tot_num_video
num_segs_every_video=kdg.num_segs_every_video #2s 세그먼트
#모든 비디오의 총 세그 수
tot_num_segs=sum(num_segs_every_video)
num_tile_per_seg=kdg.num_tile_per_seg
num_ver_per_tile=kdg.num_ver_per_tile
num_ver_per_video=num_tile_per_seg*num_ver_per_tile
#모든 버전 파일 수량 계산
tot_num_vers=sum(num_segs_every_video)*num_tile_per_seg*num_ver_per_tile

seg_popularity=kdg.getSegsPopularityList(np.arange(num_video))

#request 수 생성
num_period=24
num_sample=24
segPopularityset = kdg.generate_video_rank_train_data(num_sample)

p_per_seg=np.zeros(tot_num_segs,dtype=float)




for period in range(num_period):
    req1_cnt = 0
    for seg in range(tot_num_segs):
        # train_p_samples[period] --> sample no
        p_per_seg[seg]+=(segPopularityset[period][seg])
sum_val=sum(p_per_seg)
for seg in range(tot_num_segs):
    # train_p_samples[period] --> sample no
    p_per_seg[seg] =p_per_seg[seg]/sum_val





#bandwidth_d=kdg.getBandwidthDistribution()
num_bw_class=30
bandwidth_class=kdg.getTrainBandwidthClass(num_bw_class,13,2)

test_weight_sum=0

print('expected qoe calculating file')
#모든 버전 파일 수량 계산
tot_num_vers=sum(num_segs_every_video)*num_tile_per_seg*num_ver_per_tile




print('generate the viewports')
#expected_roi_popularity=[0.7]
#[0,30,40,80]  [10,50,90]  [20,60,70,100]  [30,70]  [20,30,60,100]
seg_expend_space_limits =[20,60,70,100] #[20,30,60,100] #  [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
vp_tiles_list = []
vp_bitmap = []
# 세그먼트마다 미리 viewport생성
for i in range(tot_num_segs):
    vp_tiles = vdg.viewportDataGenerator(num_tile_per_seg, tile_popularity[i],
                                         len(bandwidth_class))
    bitmap = []

    for r in range(len(bandwidth_class)):
        bitmap_per_request = []
        for j in range(num_tile_per_seg):
            if j in vp_tiles[r]:
                bitmap_per_request.append(1)
            else:
                bitmap_per_request.append(0)
        bitmap.append(bitmap_per_request)
    vp_bitmap.append(bitmap)
    vp_tiles_list.append(vp_tiles)


print('expected qoe calculating')
num_ver_per_seg=kdg.num_ver_per_seg
for i in range(len(seg_expend_space_limits)):

    with open('./sb3stateinfo/'+str(seg_expend_space_limits[i]) + 'state.txt', 'r') as f_state:
        seg_no=0
        while(1):
            per_seg_file_size = 0
            seg_state_str=f_state.readline()
            if(seg_state_str=='' or seg_no>=tot_num_segs):
                break
            seg_state=seg_state_str.split(' ')
            seg_start_in_ver = seg_no * num_ver_per_seg
            seg_end_in_ver = seg_start_in_ver + num_ver_per_seg
            temp_state=[]
            for j in range(num_ver_per_seg):
                temp_state.append(int(seg_state[j]))
            price=0


            core_server_request = service_train_ver2.service_train(np.full(num_ver_per_seg, 1), num_tile_per_seg,
                                                                   num_ver_per_tile,
                                                                   bitrate[seg_start_in_ver:seg_end_in_ver],
                                                                   q_list[seg_start_in_ver:seg_end_in_ver],
                                                                   vp_tiles_list[seg_no], vp_bitmap[seg_no],
                                                                   bandwidth_class)

            seg_qoe_price = service_train_ver2.service_train_QBver(temp_state, num_tile_per_seg,
                                                                  num_ver_per_tile,
                                                                  bitrate[seg_start_in_ver:seg_end_in_ver],
                                                                  q_list[seg_start_in_ver:seg_end_in_ver],
                                                                  vp_tiles_list[seg_no], vp_bitmap[seg_no],
                                                                  bandwidth_class, core_server_request)

            price += seg_qoe_price
            #print(price)
            expected_qoe=price*p_per_seg[seg_no]*10

            seg_no += 1
            #exit(2)
            if(seg_no%500==0):
                print('ratio %.2f : %d'%(seg_expend_space_limits[i],seg_no))
            with open('./sb3stateinfo/' + str(int(seg_expend_space_limits[i])) + 'seg_reward.txt', 'a') as f_reward:
                f_reward.write(str(expected_qoe))
                f_reward.write(' ')




