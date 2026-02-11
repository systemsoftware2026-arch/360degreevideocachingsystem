import statistics

import datagenerator as kdg
import numpy as np
import service
import gc
import queue
import viewportdatagenerator as vdg
import service_train_ver2
np.random.seed(1)
print('seg_storage_greedy')
file_size, bitrate, q_list=kdg.generateData()
tot_file_size=sum(file_size)
tile_popularity_list=[]
alpha=service_train_ver2.alpha
print('alpha : ', alpha)
for tp_idx in range(3):
    tile_popularity=kdg.getTilesPopularity(tp_idx)
    tile_popularity_list.append(tile_popularity)
# tile_p_list = kdg.getTilesPopularity()
# tile_popularity=kdg.getTileP_list(tile_p_list)
num_video=kdg.tot_num_video

num_segs_every_video=kdg.num_segs_every_video #2s 세그먼트
#모든 비디오의 총 세그 수
tot_num_segs=sum(num_segs_every_video)
num_tile_per_seg=kdg.num_tile_per_seg
num_ver_per_tile=kdg.num_ver_per_tile
num_ver_per_seg=kdg.num_ver_per_seg
num_ver_per_video=num_tile_per_seg*num_ver_per_tile
#모든 버전 파일 수량 계산
tot_num_vers=sum(num_segs_every_video)*num_tile_per_seg*num_ver_per_tile


#request 수 생성
num_period=24
num_sample=24
segPopularityset = kdg.getRealDatasetRankData()#generate_video_rank_train_data(num_sample)
#exit(1)
num_req_per_seg=np.zeros(tot_num_segs)
tot_num_reqs=0#sum(num_req_per_seg)
#num_req_per_period=10000
num_req1=0
req0_cnt=0
num_req_per_period=5000
# num_req1=0
# req0_cnt=0
# for period in range(num_period):
#     req0_cnt = 0
#     for seg in range(tot_num_segs):
#         # train_p_samples[period] --> sample no
#         num_req_for_seg=int(num_req_per_period * segPopularityset[period][seg])
#         if(num_req_for_seg<1 ):
#             req0_cnt+=1
#             #tot_num_reqs+=1
#             if(period<num_period):
#                 num_req_for_seg=1
#         num_req_per_seg[seg]+=(num_req_for_seg)
#         tot_num_reqs+=num_req_for_seg
#
#
#         # if(num_req_per_seg[seg]<2 and num_req_per_seg[seg]>=1):
#         #     req1_cnt+=1
#         #
#         # if(num_req_per_seg[seg]<1):
#         #      req0_cnt+=1
#     print('num_req', sum(num_req_per_seg))
# #print('req1_cnt',req1_cnt)
# print('req0_cnt',req0_cnt)
# print('req0_cnt',req0_cnt)




#[0,20,40,60,80,100]#[0,10,20,30,40,50,60,70,80,90,100]#[0,20,40,60,80,100]
seg_expend_space_limits=[0,20,30,40,50,60,70,80,90,100]#[0,20,40,60,80,100]#[0,5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100]
most_space_idx=len(seg_expend_space_limits)-1
seg_file_size=[]
seg_rewards=[]

for i in range(len(seg_expend_space_limits)):
    sub_seg_file_size=[]
    sub_seg_reward=[]
    with open('./sb3stateinfo/'+str(seg_expend_space_limits[i]) + 'state.txt', 'r') as f_state:
        seg_no=0
        while(1):
            per_seg_file_size = 0
            seg_state_str=f_state.readline()
            if(seg_state_str=='' or seg_no>=tot_num_segs):
                break
            seg_state=seg_state_str.split(' ')
            seg_start_in_ver=seg_no*num_ver_per_seg
            seg_end_in_ver=seg_start_in_ver+num_ver_per_seg
            tmp_file_size=file_size[seg_start_in_ver:seg_end_in_ver]
            pre_tile=-1
            weighted_qoe=0

            for j in range(num_ver_per_seg):
                #base버전을 제외한 저장 공간 계산
                #if(j%num_ver_per_tile!=num_ver_per_tile-1):
                    per_seg_file_size += (file_size[seg_start_in_ver+j] * float(seg_state[j]))
                    if(int(seg_state[j])==1):

                        ver_idx=seg_start_in_ver+j

                        pre_tile=j//num_ver_per_tile

            seg_no+=1
            sub_seg_file_size.append(per_seg_file_size)

    with open('./sb3stateinfo/'+str(seg_expend_space_limits[i]) + 'seg_reward.txt', 'r') as f_reward:
        while(1):
            tmp_seg_reward=f_reward.readline()
            if(tmp_seg_reward==''):
                break
            tmp_seg_reward=tmp_seg_reward.split(' ')
            for j in range(tot_num_segs):
                sub_seg_reward.append(float(tmp_seg_reward[j]))

    seg_file_size.append(sub_seg_file_size)
    seg_rewards.append(sub_seg_reward)




print(len(seg_file_size[1]))
print(len(seg_rewards[1]))

#최적화전의 초기화
selected_list=np.full(tot_num_segs,most_space_idx)
expend_space_sum=0

for seg in range(tot_num_segs):
    max_idx=most_space_idx
    max_val=seg_rewards[most_space_idx][seg]
    for j in range(most_space_idx):
        if(seg_rewards[j][seg]>max_val):
            max_idx=j
    selected_list[seg]=max_idx#most_space_idx
    expend_space_sum+=seg_file_size[max_idx][seg]
print('space sum : %.4f'%(expend_space_sum))
print(selected_list)

print('가장 높은 버전과 가장 낮은 버전의 총 용량을 계산한다.')
base_space=0
# for i in range(0, len(file_size), num_ver_per_tile):
#     #highest_ver_idx=i
#     lowest_ver_idx = i + num_ver_per_tile - 1
#     base_space+=file_size[lowest_ver_idx]
#     #base_space += file_size[highest_ver_idx]
extend_space_limit_rate=0.3

space_limit=(tot_file_size)*extend_space_limit_rate-base_space
print('space limit rate : %.2f'%(extend_space_limit_rate))
print('space_limit : %.4f'%(space_limit))
print('base space : %.4f'%(base_space))
seg_h_list=[]
tmp_file_size=0

seg_h_queue=queue.PriorityQueue()
print('휴리스틱 값 계산 중')
scale_factor=1
for i in range(tot_num_segs):
    per_seg_h_list=[]
    for j in range(most_space_idx):
        if((seg_file_size[most_space_idx][i]-seg_file_size[j][i])==0 or j>=selected_list[i]):
            continue
        qoe_diff=(seg_rewards[selected_list[i]][i]-seg_rewards[j][i])*scale_factor
        seg_file_size_diff=((seg_file_size[selected_list[i]][i] - seg_file_size[j][i]))
        if(seg_file_size_diff==0):
            seg_file_size_diff=0.000001
        h_val=qoe_diff/seg_file_size_diff

        per_seg_h_list.append([h_val,i,j])
        
    per_seg_h_list.sort(key=lambda per_seg_h_list: per_seg_h_list[0],reverse=True)
    if per_seg_h_list:  # 列表非空时才pop
        seg_h = per_seg_h_list.pop()
        #seg_h=per_seg_h_list.pop()
        seg_h_queue.put(seg_h)
    seg_h_list.append(per_seg_h_list)

canSelect=np.full(tot_num_segs,1)

num_bwclass=30
print('휴리스틱 알고리즘 작동 중')
cnt=0
while(seg_h_queue.empty()!=True):
    seg_h=seg_h_queue.get()

    seg_no = seg_h[1]
    select = seg_h[2]
    if(selected_list[seg_no]>select and canSelect[seg_no]==1):#and len(seg_expend_space_limits)>10
        selected = selected_list[seg_no]
        # if(seg_no>tot_num_segs*0.9):
        #     select=0
        expend_space_sum = expend_space_sum - seg_file_size[selected][seg_no] + seg_file_size[select][seg_no]
        if (expend_space_sum <= space_limit):
            break
        selected_list[seg_no] = select


    h_list_for_seg = []
    for i in range(len(seg_h_list[seg_no])):
        qoe_diff =   (
                seg_rewards[select][seg_no] - seg_rewards[seg_h_list[seg_no][i][2]][seg_h_list[seg_no][i][1]])*scale_factor
        seg_file_size_diff = seg_file_size[select][seg_no] - seg_file_size[seg_h_list[seg_no][i][2]][
            seg_h_list[seg_no][i][1]]
        if (seg_file_size_diff == 0):
            seg_file_size_diff = 0.000001
        h_val = qoe_diff / seg_file_size_diff
        seg_h_list[seg_no][i][0] = h_val
    h_list_for_seg = seg_h_list[seg_no]

    if (len(h_list_for_seg) > 1):
        h_list_for_seg.sort(key=lambda h_list_for_seg: h_list_for_seg[0], reverse=True)
        seg_h_list[seg_no] = h_list_for_seg
        seg_h = seg_h_list[seg_no].pop()
        seg_h_queue.put(seg_h)
    elif (len(h_list_for_seg) == 1):
        if (canSelect[seg_no] == 1):
            #print('test')
            seg_h = seg_h_list[seg_no][0]
            seg_h_queue.put(seg_h)
            canSelect[seg_no] = 0


    cnt+=1
    if(cnt%40000==0):
        print('len seg_h_queue : %d'%(seg_h_queue.qsize()))


print('seg_h_list length : %d'%(len(seg_h_list)))
final_size_sum=0
del seg_h_list



state=np.full(tot_num_vers,0)
num_vers_storage=np.zeros(num_ver_per_tile)
print('전체 state 구성 중')
seg_select_list=[]
for i in range(len(seg_expend_space_limits)):
    seg_select_list.append([])
for i in range(tot_num_segs):
    select = selected_list[i]
    seg_select_list[select].append(i)
for i in range(len(seg_select_list)):
    print('select : %d seg_cnt : %d' % (i,len(seg_select_list[i])))

for select in range(len(seg_expend_space_limits)):
    if(len(seg_select_list[select])==0):
        continue
    idx=0
    print('select : %d , num_seg per select : %d'%(select,len(seg_select_list[select])))
    with open('./sb3stateinfo/' + str(seg_expend_space_limits[select]) + 'state.txt', 'r') as f_state:
        seg_cnt=0
        seg_state_str=''
        while(1):
            seg_state_str = f_state.readline()
            if(seg_cnt%40000==0):
                print('seg_cnt : %d'%(seg_cnt))
            if (seg_state_str == ''):
                break
            if (seg_cnt == seg_select_list[select][idx]):

                seg_state = seg_state_str.split(' ')
                for j in range(num_ver_per_seg):
                    seg_state[j] = int(seg_state[j])
                seg_start_in_ver = seg_cnt * num_ver_per_seg
                seg_end_in_ver = seg_start_in_ver + num_ver_per_seg
                state[seg_start_in_ver:seg_end_in_ver] = seg_state
                for j in range(num_ver_per_seg):
                    if (state[seg_start_in_ver + j] == 1):
                        num_vers_storage[int((seg_start_in_ver + j) % num_ver_per_tile)] += 1
                final_size_sum += seg_file_size[select][seg_cnt]
                idx += 1
                if(idx>=len(seg_select_list[select])):
                    break
            seg_cnt+=1

print('-------------------------------------result----------------------------------------------')
theta=int(kdg.global_theta*10)
file_path = './txtdata/requests_log'+str(theta)+('_real1.txt')
final_qoe=0
final_backhaul_bw=0
final_req_qoe=0
tmp_benefit=0
tmp_backhaul_bw=0
tmp_req_bw=0
beta=1
with open(file_path, 'r', encoding='utf-8') as file:
    line_cnt=0
    second_cnt=0
    for second, line in enumerate(file, start=1):
        if(line_cnt%600!=0):
            line_cnt+=1
            continue
        # if(line_cnt>20):
        #     print('test 끝났음')
        #     exit(3)
        # 读取行
        line_content = line.strip()
        #print(f"Raw line for second {second}: {line_content}")

        # 分割行内元素
        seg_numbers = line_content.split(' ')  # 假设seg号用逗号分隔    `
        for req_idx in range(len(seg_numbers)):
            seg_numbers[req_idx]=int(seg_numbers[req_idx])
        #len(seg_numbers)
        bandwidth_d=bandwidth_class=kdg.sample_logn_trunc(len(seg_numbers), mu_log=2.346369415862733, sigma_log=0.3114876058248091, tau=7.0, seed=1)
        #kdg.getTrainBandwidthClass(len(seg_numbers))
        tile_popularity=tile_popularity_list[int(second_cnt//3600%3)]
        vp_tiles_list, vp_bitmap=service.viewport_data_generate(len(seg_numbers),tile_popularity,seg_numbers)

        print(f'Processed seg numbers for second {line_cnt}')
        core_server_request = service.service2(num_tile_per_seg,
                                               num_ver_per_tile,
                                               bitrate,
                                               q_list,
                                               vp_tiles_list, vp_bitmap,
                                               bandwidth_d, seg_numbers)
        #print(state)
        # if(line_cnt>1):
        #     exit(5)


        mean_backhaul_bw,mean_req_bw,hit_ratio,mean_service_qoe,mean_request_qoe,backhaul_bw=service_train_ver2.service_QB_for_Algo(state,num_tile_per_seg,num_ver_per_tile,bitrate,q_list,vp_bitmap,
                                                                                         vp_tiles_list,bandwidth_d,core_server_request,seg_numbers,beta)
        line_cnt += 1
        second_cnt+=1
        tmp_backhaul_bw+=mean_backhaul_bw
        tmp_req_bw+=mean_req_bw

        final_backhaul_bw=tmp_backhaul_bw/second_cnt
        final_req_bw=tmp_req_bw/second_cnt
        final_qoe+=mean_service_qoe
        final_req_qoe+=mean_request_qoe
        final_benefit_ratio = alpha*(final_qoe/final_req_qoe)-(1-alpha)*tmp_backhaul_bw/tmp_req_bw
        print(f" num_req: {len(seg_numbers)} \nfinal mean qoe: {final_qoe/second_cnt}, final_req_qoe: {final_req_qoe/second_cnt},"
              f" mean backhaul bw: {mean_backhaul_bw} ")
        print(f"final benefit_ratio: {final_benefit_ratio}, backhaul_bw: {final_backhaul_bw} "
              f"\nmean_service_qoe: {mean_service_qoe}, mean_request_qoe: {mean_request_qoe}")
        # if(line_cnt>1):
        #     exit(5)

exit(1)



bandwidth_d=[]
for seg in range(tot_num_segs):
    b_d=kdg.getTrainBandwidthClass(num_req_per_seg[seg])
    bandwidth_d.append(b_d)
#bandwidth_d=kdg.getBandwidthDistribution()

# for seg in range(tot_num_segs):
#     num_req_per_seg[seg]=len(bandwidth_d[seg])


vp_tiles_list = []
vp_bitmap = []
# 세그먼트마다 미리 viewport생성
for i in range(tot_num_segs):
    vp_tiles = vdg.viewportDataGenerator(num_tile_per_seg, tile_popularity[i],
                                         len(bandwidth_d[i]))
    bitmap = []

    for r in range(len(bandwidth_d[i])):
        bitmap_per_request = []
        for j in range(num_tile_per_seg):
            if j in vp_tiles[r]:
                bitmap_per_request.append(1)
            else:
                bitmap_per_request.append(0)
        bitmap.append(bitmap_per_request)
    vp_bitmap.append(bitmap)
    vp_tiles_list.append(vp_tiles)
# print('vp_tiles_list[10][2]')
# print(vp_tiles_list[10][2])
#aver_qoe=service.service2(state, tot_num_segs, num_tile_per_seg, num_ver_per_tile, bandwidth_d, bitrate, q_list, vp_tiles_list,vp_bitmap)
deg_qoe_sum=0
num_req=0
hit_cnt=0
tot_backhaul_bw=0
num_tile_req=0
num_exceed_req=0
for seg in range(tot_num_segs):
    seg_start_in_ver = seg * num_ver_per_seg
    seg_end_in_ver = seg_start_in_ver + num_ver_per_seg
    #print(len(bandwidth_d[seg]))
    if(len(bandwidth_d[seg])==0):
        continue

    core_server_request = service_train_ver2.service_train(np.full(num_ver_per_seg, 1), num_tile_per_seg,
                                                           num_ver_per_tile,
                                                           bitrate[seg_start_in_ver:seg_end_in_ver],
                                                           q_list[seg_start_in_ver:seg_end_in_ver],
                                                           vp_tiles_list[seg], vp_bitmap[seg],
                                                           bandwidth_d[seg])
    core_req_qoe=0
    #print('len(core_server_request) %d'%(len(core_server_request)))
    for req_idx in range(len(core_server_request)):
        vp_qoe_list = []
        qoe=q_list[seg_start_in_ver:seg_end_in_ver]
        select_ver=core_server_request[req_idx]
        #print(len(vp_bitmap[seg]))
        for tile in range(len(vp_bitmap[seg][req_idx])):
            num_tile_req+=1
            if vp_bitmap[seg][req_idx][tile] == 1:

                vp_qoe_list.append(qoe[select_ver[tile]])
        #print(vp_qoe_list)
        vp_stdev = statistics.stdev(vp_qoe_list)
        vp_mean=statistics.mean(vp_qoe_list)
        core_req_qoe+=(vp_mean-vp_stdev)

    seg_ratio,seg_backhaul_bw = service_train_ver2.service_QB_for_Algo(state[seg_start_in_ver:seg_end_in_ver], num_tile_per_seg,
                                                          num_ver_per_tile,
                                                          bitrate[seg_start_in_ver:seg_end_in_ver],
                                                          q_list[seg_start_in_ver:seg_end_in_ver],
                                                          vp_tiles_list[seg], vp_bitmap[seg],
                                                          bandwidth_d[seg],core_server_request)
    if(seg_backhaul_bw>3):
        num_exceed_req+=1
    deg_seg_qoe=core_req_qoe*(1-seg_ratio)/len(core_server_request)
    deg_qoe_sum+=(deg_seg_qoe*(num_req_per_seg[seg]/tot_num_reqs))
    tot_backhaul_bw+=(seg_backhaul_bw*(num_req_per_seg[seg]/tot_num_reqs))
    #num_req+=len(bandwidth_d[seg])
    if(seg%1000==0):
        ##print('seg : %d : %.4f , %.4f'%(seg,seg_qoe,qoe_sum/10))
        print('seg : %d deg_seg_qoe : %.4f , deg_qoe_sum : %.4f, seg_backhaul_bw : %.4f '
              'tot_backhaul_bw: %.4f'%(seg,deg_seg_qoe,deg_qoe_sum,seg_backhaul_bw,tot_backhaul_bw))
aver_qoe=deg_qoe_sum#/num_req#tot_num_segs
# if(num_service_denial>0):
#     aver_qoe=qoe_sum*(tot_num_reqs-num_service_denial)/tot_num_reqs
#
# print('num_service_denial : %d' % (num_service_denial))
print('aver enhance qoe : %.4f'%(aver_qoe))
print('aver backhaul bw :  %.4f'%(tot_backhaul_bw))
print('num_exceed_req :  %d'%(num_exceed_req))
print(selected_list[:100])
print('space limit rate : %.2f'%(extend_space_limit_rate))
