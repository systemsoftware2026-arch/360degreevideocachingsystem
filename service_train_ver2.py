import math
import random

import numpy as np

import statistics
alpha=0.7
def service_QB_for_Algo(state,num_tile_per_seg,num_ver_per_tile,bitrate_,qoe_,vp_bitmap_list,vp_tiles_seg,bandwidth_class,requests_ver,seg_numbers,beta):
    num_ver_per_seg = num_tile_per_seg * num_ver_per_tile
    tot_qoe = 0
    beta=0.3
    qoe_per_seg = 0
    num_request_per_seg = len(seg_numbers)  # num_bandwidth
    viewport_qoe = 0
    num_ver_service = np.zeros(num_ver_per_tile)
    num_vp_tiles = 0
    num_hmd_tile = 9  # 3
    num_viewport_tile = 0
    min_max_diff_qoe_sum = 0
    vp_stdev_sum = 0
    service_qoe_ratio_sum = 0
    backhaul_bw_sum=0
    select_ver_list=[]
    backhaul_ver_list=[]
    tot_req=num_tile_per_seg*num_request_per_seg
    hit_cnt=0
    miss_cnt=0
    for i in range(num_request_per_seg):
        seg=requests_ver[i][0]
        vp_bitmap = vp_bitmap_list[i]
        vp_tiles = vp_tiles_seg[i]
        bw_limit = bandwidth_class[i]
        vp_heuristic_values_table = []
        request_ver = requests_ver[i][1]
        request_bw = 0
        # for k in range(len(request_ver)):
        #     request_bw += bitrate[request_ver[k]]
        # if (request_bw > bw_limit):
        #     print('request bw : %.4f, bw_limit : %.4f'%(request_bw,bw_limit))
        #     exit(3)
        seg_start_in_ver = seg * num_ver_per_seg
        seg_end_in_ver = seg_start_in_ver + num_ver_per_seg
        bitrate=bitrate_[seg_start_in_ver:seg_end_in_ver]
        qoe=qoe_[seg_start_in_ver:seg_end_in_ver]
        select_ver = np.full(num_tile_per_seg, 0)
        backhaul_bw = 0
        backhaul_vers=np.full(num_tile_per_seg,-1).tolist()
        # client bandwidth

        bw_sum = 0

        tile_no = 0

        # request가 요청한 버전들을 캐쉬에 저장되어 있는지 확인하고 없으면 degradation
        for tile_no in range(num_tile_per_seg):
            ver_no = request_ver[tile_no]
            #print('test3',ver_no)
            # print(ver_no)
            if state[ver_no+seg_start_in_ver] == 1:
                hit_cnt+=1
                select_ver[tile_no] = ver_no
                backhaul_vers[tile_no]=ver_no
            else:
                tile_end_in_ver = tile_no * num_ver_per_tile + num_ver_per_tile
                isVirtualHit = False
                for tmp_idx in range(ver_no+1, tile_end_in_ver):
                    if state[tmp_idx+seg_start_in_ver] == 1:
                        select_ver[tile_no] = tmp_idx
                        backhaul_vers[tile_no]=tmp_idx
                        isVirtualHit = True
                        hit_cnt+=1
                        break
                #miss
                if (isVirtualHit == False):
                    miss_cnt+=1
                    select_ver[tile_no] = ver_no
                    backhaul_vers[tile_no]=-1
                    backhaul_bw_sum+=bitrate[select_ver[tile_no]]


        select_ver_list.append([i,select_ver])
        backhaul_ver_list.append([i,backhaul_vers])

    print(f"backhaul_bw_sum test : {backhaul_bw_sum}")


    # num_client_mu=2000
    # num_client_sigma=200
    #1Gbps-10Gbps
    backhaul_bw_gap = 600
    # select_ver_list,backhaul_bw_sum = server_backhall_ver_decision(backhaul_ver_list, select_ver_list, backhaul_bw_gap, bitrate_, qoe_,
    #                                                   num_tile_per_seg, num_ver_per_tile)
    tmp_backhaul_bw_sum=0
    request_qoe_sum=0
    service_qoe_sum=0
    bw_limit_sum=0
    current_bk_latency=0
    tot_current_bk_latency=0
    latency_sum=0
    pre_latency=0
    pre_backhaul_bw=0
    for i in range(num_request_per_seg):
        backhaul_bw = 0
        select_ver = select_ver_list[i][1]
        seg = requests_ver[i][0]
        seg_start_in_ver = seg * num_ver_per_seg
        seg_end_in_ver = seg_start_in_ver + num_ver_per_seg
        bitrate = bitrate_[seg_start_in_ver:seg_end_in_ver]
        qoe = qoe_[seg_start_in_ver:seg_end_in_ver]
        pre_latency=tmp_backhaul_bw_sum / backhaul_bw_gap
        #print('tmp_backhaul_bw_sum',tmp_backhaul_bw_sum)
        #print('current_bk_latency: ',current_bk_latency)
        for tile in range(num_tile_per_seg):
            if (backhaul_ver_list[i][1][tile] != -1):
                continue
            ver_no = select_ver[tile]
            backhaul_bw+= bitrate[ver_no]
        vp_bitmap = vp_bitmap_list[i]
        request_ver = requests_ver[i][1]
        # for k in range(len(select_ver)):
        #     if vp_bitmap[k] == 1:
        #         print(qoe[request_ver[k]],end=' ')
        #print('ver 2 req ',i,' vp_bitmap',vp_bitmap)
        #print('core request: ',request_ver)
        #print('latency')
        bw_limit = bandwidth_class[i]
        pre_backhaul_bw=backhaul_bw
        tmp_backhaul_bw_sum+=backhaul_bw
        bw_limit_sum+=bw_limit

        vp_bitrate = 0
        vp_qoe = 0
        deliver_bitrate_sum = 0
        not_vp_bitrate = 0

        vp_stdev = 0
        vp_qoe_list = []
        request_vp_qoe_list = []
        for k in range(len(select_ver)):
            deliver_bitrate_sum += bitrate[request_ver[k]]
            if vp_bitmap[k] == 1:
                num_viewport_tile += 1
                vp_bitrate += bitrate[select_ver[k]]
                vp_qoe += qoe[select_ver[k]]
                vp_qoe_list.append(qoe[select_ver[k]])
                request_vp_qoe_list.append(qoe[request_ver[k]])
                viewport_qoe += qoe[select_ver[k]]
                num_ver_service[select_ver[k] % num_ver_per_tile] += 1
            else:
                not_vp_bitrate += bitrate[select_ver[k]]
            tot_qoe += qoe[select_ver[k]]
        vp_stdev = statistics.stdev(vp_qoe_list)
        vp_mean = statistics.mean(vp_qoe_list)
        service_qoe = vp_mean - vp_stdev
        #print('service_qoe',service_qoe)
        latency_sum+=current_bk_latency
        service_qoe_sum+=(service_qoe-0*current_bk_latency)

        request_vp_stddev = statistics.stdev(request_vp_qoe_list)
        request_vp_mean = statistics.mean(request_vp_qoe_list)
        request_qoe = request_vp_mean - request_vp_stddev
        #print('request_qoe',request_qoe,request_vp_qoe_list,' request bitrate:',deliver_bitrate_sum)
        request_qoe_sum+=request_qoe
        #service_qoe_ratio = service_qoe / request_qoe

        #service_qoe_ratio_sum += service_qoe_ratio - backhaul_bw_ratio
        vp_stdev_sum += vp_stdev

    # if(backhaul_bw_sum>backhaul_bw_gap):
    #     backhaul_bw_sum=backhaul_bw_gap
    latency=current_bk_latency/backhaul_bw_gap
    mean_service_qoe=service_qoe_sum/num_request_per_seg
    mean_request_qoe=request_qoe_sum/num_request_per_seg

    tmp_backhaul_bw_sum=max(0.01,tmp_backhaul_bw_sum/num_request_per_seg)
    service_benefit_ratio_sum=alpha*(mean_service_qoe/mean_request_qoe)-(1-alpha)*(tmp_backhaul_bw_sum/bw_limit_sum)
    hit_ratio=hit_cnt/tot_req
    #print('tmp_backhaul_bw_sum',tmp_backhaul_bw_sum)
    print('miss cnt',miss_cnt,'mean_service_qoe: ',mean_service_qoe,' mean_request_qoe: ',mean_request_qoe,'current_bk_latency',current_bk_latency)
    return backhaul_bw_sum/num_request_per_seg,bw_limit_sum/num_request_per_seg,hit_ratio,mean_service_qoe,mean_request_qoe,tmp_backhaul_bw_sum # aver_vp_qoe


def service_train(state,num_tile_per_seg,num_ver_per_tile,bitrate,qoe,vp_tiles_seg,vp_bitmap_list,bandwidth_class):
    num_ver_per_seg = num_tile_per_seg * num_ver_per_tile
    tot_qoe=0


    qoe_per_seg = 0
    num_request_per_seg = len(bandwidth_class)#num_bandwidth
    viewport_qoe=0
    num_ver_service = np.zeros(num_ver_per_tile)
    num_vp_tiles=0
    num_hmd_tile = 9  #3
    num_viewport_tile=0
    min_max_diff_qoe_sum=0
    vp_stdev_sum=0
    select_ver_list=[]
    for i in range(num_request_per_seg):
        vp_bitmap=vp_bitmap_list[i]
        #print(vp_bitmap)
        vp_tiles=vp_tiles_seg[i]
        vp_heuristic_values_table=[]

        select_ver = np.full(num_tile_per_seg, 0)
        # client bandwidth
        bw_limit = bandwidth_class[i]
        bw_sum = 0
        seg_start_in_ver = 0 * num_ver_per_seg
        seg_end_in_ver = seg_start_in_ver + num_ver_per_seg
        # 해당 seg의 버전은 다 highest버전으로 선택한다.
        tile_no = 0
        for k in range(seg_start_in_ver, seg_end_in_ver, num_ver_per_tile):
            highest_ver_idx = k
            lowest_ver_idx=k+num_ver_per_tile-1
            for tmp_idx in range(k,k+num_ver_per_tile):
                if(state[tmp_idx]==1):
                    highest_ver_idx=tmp_idx
                    break
            if(vp_bitmap[tile_no]==1):
                select_ver[tile_no] = highest_ver_idx

                bw_sum += bitrate[highest_ver_idx]
                num_extend_ver=0
                second_ver=0
                extend_ver_list = []
                #휴리스틱 값의 표 생성, viewport tile만 계산
                for v in range(highest_ver_idx + 1, lowest_ver_idx + 1):
                    if (state[v] == 1):
                        qoe_diff = qoe[highest_ver_idx] - qoe[v]
                        bitrate_diff = bitrate[highest_ver_idx] - bitrate[v]
                        if(bitrate_diff==0):
                            bitrate_diff=1e-12
                        num_extend_ver += 1
                        h_val=qoe_diff  / bitrate_diff
                        extend_ver_list.append([v, h_val])
                extend_ver_list.sort(key=lambda extend_ver_list: extend_ver_list[1])
                if (num_extend_ver > 0):
                    vp_heuristic_values_table.append([tile_no, extend_ver_list[0][1], extend_ver_list])

            else:
                select_ver[tile_no] = lowest_ver_idx
                bw_sum += bitrate[lowest_ver_idx]
            tile_no += 1
        # print(bw_sum)
        # print(bw_limit)
        if (bw_sum > bw_limit):

            # 휴리스틱 값에 따라 오름 차순으로 정렬한다.
            vp_heuristic_values_table.sort(key=lambda vp_heuristic_values_table: vp_heuristic_values_table[1])

            #비 viewport 버전 다 낮춰도 대역폭은 limit보다 더 크면 viewport 버전도 낮춰야 된다.
            if (bw_sum > bw_limit):
                while (len(vp_heuristic_values_table) != 0):
                    ver_idx = vp_heuristic_values_table[0][2][0][0]  # 체크할 버전 파일의 인덱스
                    tile_no = vp_heuristic_values_table[0][0]  # select_ver의 index를 구한다.

                    del vp_heuristic_values_table[0][2][0]
                    if (ver_idx > select_ver[tile_no]):
                        bw_sum = bw_sum - bitrate[select_ver[tile_no]] + bitrate[ver_idx]
                        select_ver[tile_no] = ver_idx

                    if (bw_sum > bw_limit):
                        if (len(vp_heuristic_values_table[0][2]) > 0):
                            extend_ver_list = []
                            for tmp_idx in range(len(vp_heuristic_values_table[0][2])):
                                tmp_ver = vp_heuristic_values_table[0][2][tmp_idx][0]
                                if (tmp_ver <= select_ver[tile_no]):
                                    continue
                                qoe_diff = qoe[select_ver[tile_no]] - qoe[tmp_ver]
                                bitrate_diff = bitrate[select_ver[tile_no]] - bitrate[tmp_ver]
                                if (bitrate_diff == 0):
                                    bitrate_diff = 1e-12
                                vp_heuristic_values_table[0][2][tmp_idx][1] = qoe_diff  / bitrate_diff
                                extend_ver_list = vp_heuristic_values_table[0][2]
                            extend_ver_list.sort(key=lambda extend_ver_list: extend_ver_list[1])

                            if(len(extend_ver_list)>0):
                                vp_heuristic_values_table[0][2] = extend_ver_list
                                vp_heuristic_values_table[0][1] = vp_heuristic_values_table[0][2][0][1]

                                vp_heuristic_values_table.sort(
                                    key=lambda vp_heuristic_values_table: vp_heuristic_values_table[1])
                            else:
                                del vp_heuristic_values_table[0]
                        else:
                            del vp_heuristic_values_table[0]
                    else:
                        break

        vp_bitrate = 0
        vp_qoe = 0
        deliver_bitrate_sum = 0
        not_vp_bitrate = 0
        vp_tile_cnt = 0
        vp_max_qoe = 0
        vp_min_qoe = 100
        vp_stdev=0
        vp_qoe_list=[]
        for k in range(len(select_ver)):
            deliver_bitrate_sum += bitrate[select_ver[k]]
            if vp_bitmap[k] == 1:
                num_viewport_tile += 1
                vp_bitrate += bitrate[select_ver[k]]
                vp_qoe += qoe[select_ver[k]]
                vp_qoe_list.append(qoe[select_ver[k]])
                viewport_qoe += qoe[select_ver[k]]
                num_ver_service[select_ver[k] % num_ver_per_tile] += 1
            else:
                not_vp_bitrate += bitrate[select_ver[k]]
            tot_qoe += qoe[select_ver[k]]
        vp_stdev=statistics.stdev(vp_qoe_list)
        vp_stdev_sum+=vp_stdev
        min_max_diff_qoe_sum+=(vp_max_qoe-vp_min_qoe)

        if (deliver_bitrate_sum > bw_limit):
            print('vp_bitrate : %.2f not_vp_bitrate : %.2f deliver_bitrate_sum : %.2f bw_limit %.2f vp_qoe : %.2f' % (
            vp_bitrate, not_vp_bitrate, deliver_bitrate_sum, bw_limit, vp_qoe / num_hmd_tile))

            exit(1)
        select_ver_list.append(np.copy(select_ver).tolist())
    aver_qoe=tot_qoe/num_tile_per_seg/num_request_per_seg
    min_max_qoe_mean=min_max_diff_qoe_sum/num_request_per_seg
    vp_stdev_mean=vp_stdev_sum/num_request_per_seg
    aver_vp_qoe=viewport_qoe/num_viewport_tile-vp_stdev_mean
    return select_ver_list

# def service_train_QBver(state,num_tile_per_seg,num_ver_per_tile,bitrate,qoe,vp_tiles_seg,vp_bitmap_list,bandwidth_class,requests_ver):
#     num_ver_per_seg = num_tile_per_seg * num_ver_per_tile
#     tot_qoe=0
#
#
#     qoe_per_seg = 0
#     num_request_per_seg = len(bandwidth_class)#num_bandwidth
#     viewport_qoe=0
#     num_ver_service = np.zeros(num_ver_per_tile)
#     num_vp_tiles=0
#     num_hmd_tile = 9  #3
#     num_viewport_tile=0
#     min_max_diff_qoe_sum=0
#     vp_stdev_sum=0
#     service_qoe_ratio_sum=0
#     for i in range(num_request_per_seg):
#
#
#         vp_bitmap=vp_bitmap_list[i]
#         vp_tiles=vp_tiles_seg[i]
#         bw_limit = bandwidth_class[i]
#         vp_heuristic_values_table=[]
#         request_ver=requests_ver[i]
#         request_bw=0
#         # for k in range(len(request_ver)):
#         #     request_bw += bitrate[request_ver[k]]
#         # if (request_bw > bw_limit):
#         #     print('request bw : %.4f, bw_limit : %.4f'%(request_bw,bw_limit))
#         #     exit(3)
#
#
#
#         select_ver = np.full(num_tile_per_seg, 0)
#         backhaul_bw=0
#         # client bandwidth
#
#         bw_sum = 0
#         seg_start_in_ver = 0 * num_ver_per_seg
#         seg_end_in_ver = seg_start_in_ver + num_ver_per_seg
#         tile_no = 0
#
#         #request가 요청한 버전들을 캐쉬에 저장되어 있는지 확인하고 없으면 degradation
#         for tile_no in range(num_tile_per_seg):
#             ver_no=request_ver[tile_no]
#             #print(ver_no)
#             if state[ver_no]==1:
#                 select_ver[tile_no]=ver_no
#             else:
#                 tile_end_in_ver=seg_start_in_ver+tile_no*num_ver_per_tile+num_ver_per_tile
#                 isVirtualHit=False
#                 for tmp_idx in range(ver_no,tile_end_in_ver):
#                     if state[tmp_idx] == 1:
#                         select_ver[tile_no] = tmp_idx
#                         isVirtualHit=True
#                         break
#                 if(isVirtualHit==False):
#                     select_ver[tile_no]=ver_no
#                     backhaul_bw+=bitrate[ver_no]
#         backhaul_bw_ratio=backhaul_bw/bw_limit
#         vp_bitrate = 0
#         vp_qoe = 0
#         deliver_bitrate_sum = 0
#         not_vp_bitrate = 0
#         vp_tile_cnt = 0
#
#         vp_stdev=0
#         vp_qoe_list=[]
#         request_vp_qoe_list=[]
#         for k in range(len(select_ver)):
#             deliver_bitrate_sum += bitrate[select_ver[k]]
#             if vp_bitmap[k] == 1:
#                 num_viewport_tile += 1
#                 vp_bitrate += bitrate[select_ver[k]]
#                 vp_qoe += qoe[select_ver[k]]
#                 vp_qoe_list.append(qoe[select_ver[k]])
#                 request_vp_qoe_list.append(qoe[request_ver[k]])
#                 viewport_qoe += qoe[select_ver[k]]
#                 num_ver_service[select_ver[k] % num_ver_per_tile] += 1
#             else:
#                 not_vp_bitrate += bitrate[select_ver[k]]
#             tot_qoe += qoe[select_ver[k]]
#         vp_stdev=statistics.stdev(vp_qoe_list)
#         vp_mean=statistics.mean(vp_qoe_list)
#         service_qoe=vp_mean-vp_stdev
#
#         request_vp_stdev=statistics.stdev(request_vp_qoe_list)
#         request_vp_mean=statistics.mean(request_vp_qoe_list)
#         request_qoe=request_vp_mean-request_vp_stdev
#
#         service_qoe_ratio = service_qoe/request_qoe
#
#         service_qoe_ratio_sum+=service_qoe_ratio-backhaul_bw_ratio
#         vp_stdev_sum+=vp_stdev
#
#         if (deliver_bitrate_sum > bw_limit):
#             print('vp_bitrate : %.2f not_vp_bitrate : %.2f deliver_bitrate_sum : %.2f bw_limit %.2f vp_qoe : %.2f' % (
#             vp_bitrate, not_vp_bitrate, deliver_bitrate_sum, bw_limit, vp_qoe / num_hmd_tile))
#             for k in range(len(select_ver)):
#                 if vp_bitmap[k] == 1:
#                     print(select_ver[k],end='\t')
#             exit(1)
#
#     # aver_qoe=tot_qoe/num_tile_per_seg/num_request_per_seg
#     # min_max_qoe_mean=min_max_diff_qoe_sum/num_request_per_seg
#     # vp_stdev_mean=vp_stdev_sum/num_request_per_seg
#     # aver_vp_qoe=viewport_qoe/num_viewport_tile-vp_stdev_mean
#     return service_qoe_ratio_sum/num_request_per_seg#aver_vp_qoe

def service_train_QBver(state, num_tile_per_seg, num_ver_per_tile, bitrate, qoe, vp_tiles_seg, vp_bitmap_list,
                        bandwidth_class, requests_ver):
    num_ver_per_seg = num_tile_per_seg * num_ver_per_tile
    tot_qoe = 0

    qoe_per_seg = 0
    num_request_per_seg = len(bandwidth_class)  # num_bandwidth
    viewport_qoe = 0
    num_ver_service = np.zeros(num_ver_per_tile)
    num_vp_tiles = 0
    num_hmd_tile = 9  # 3
    num_viewport_tile = 0
    min_max_diff_qoe_sum = 0
    vp_stdev_sum = 0
    service_qoe_ratio_sum = 0
    select_ver_list=[]
    backhaul_ver_list=[]
    for i in range(num_request_per_seg):

        vp_bitmap = vp_bitmap_list[i]
        vp_tiles = vp_tiles_seg[i]
        bw_limit = bandwidth_class[i]
        vp_heuristic_values_table = []
        request_ver = requests_ver[i]
        request_bw = 0
        # for k in range(len(request_ver)):
        #     request_bw += bitrate[request_ver[k]]
        # if (request_bw > bw_limit):
        #     print('request bw : %.4f, bw_limit : %.4f'%(request_bw,bw_limit))
        #     exit(3)

        select_ver = np.full(num_tile_per_seg, 0)
        backhaul_ver = np.full(num_tile_per_seg, -1)
        backhaul_bw = 0
        # client bandwidth

        bw_sum = 0
        seg_start_in_ver = 0 * num_ver_per_seg
        seg_end_in_ver = seg_start_in_ver + num_ver_per_seg
        tile_no = 0

        # request가 요청한 버전들을 캐쉬에 저장되어 있는지 확인하고 없으면 degradation
        for tile_no in range(num_tile_per_seg):
            ver_no = request_ver[tile_no]
            # print(ver_no)
            if state[ver_no] == 1:
                select_ver[tile_no] = ver_no
                backhaul_ver[tile_no] = ver_no
            else:
                tile_end_in_ver = seg_start_in_ver + tile_no * num_ver_per_tile + num_ver_per_tile
                isVirtualHit = False
                for tmp_idx in range(ver_no, tile_end_in_ver):
                    if state[tmp_idx] == 1:
                        select_ver[tile_no] = tmp_idx
                        backhaul_ver[tile_no] = tmp_idx
                        isVirtualHit = True
                        break
                if (isVirtualHit == False):
                    select_ver[tile_no] = ver_no
                    backhaul_ver[tile_no]=-1

        backhaul_ver_list.append(backhaul_ver.tolist())
        select_ver_list.append(select_ver.tolist())

    #num_client_mu=2000
    #num_client_sigma=200
    # num_client=[1600,1700,1800,1900]
    # clt_idx=random.randint(0,3)
    # backhaul_gap_list=[200,400,600,800,1000]
    # backhaul_gap_idx=random.randint(0,4)
    # backhaul_bw_gap=backhaul_gap_list[backhaul_gap_idx]/num_client[clt_idx]*num_request_per_seg
    # select_ver_list=backhall_ver_decision_for_train(backhaul_ver_list,select_ver_list,backhaul_bw_gap,bitrate,qoe,num_tile_per_seg,num_ver_per_tile)
    delta=0.2
    for i in range(num_request_per_seg):
        backhaul_bw = 0
        select_ver=select_ver_list[i]
        backhaul_ver=backhaul_ver_list[i]
        #print(backhaul_ver)
        for tile in range(num_tile_per_seg):
            ver_no=select_ver[tile]
            if(backhaul_ver[tile]==-1):
                #print(backhaul_ver[tile],end=' ')
                backhaul_bw+=bitrate[ver_no]
            #print(bitrate[ver_no])

        vp_bitmap = vp_bitmap_list[i]
        request_ver = requests_ver[i]
        bw_limit = bandwidth_class[i]
        backhaul_bw_ratio = backhaul_bw / bw_limit
        #print('backhaul_bw_ratio',backhaul_bw_ratio,'backhaul bw: ',backhaul_bw,'bw_limit: ',bw_limit)
        vp_bitrate = 0
        vp_qoe = 0
        deliver_bitrate_sum = 0
        not_vp_bitrate = 0


        vp_stdev = 0
        vp_qoe_list = []
        request_vp_qoe_list = []
        for k in range(len(select_ver)):
            deliver_bitrate_sum += bitrate[select_ver[k]]
            if vp_bitmap[k] == 1:
                num_viewport_tile += 1
                vp_bitrate += bitrate[select_ver[k]]
                vp_qoe += qoe[select_ver[k]]
                vp_qoe_list.append(qoe[select_ver[k]])
                request_vp_qoe_list.append(qoe[request_ver[k]])
                viewport_qoe += qoe[select_ver[k]]
                num_ver_service[select_ver[k] % num_ver_per_tile] += 1
            else:
                not_vp_bitrate += bitrate[select_ver[k]]
            tot_qoe += qoe[select_ver[k]]
        vp_stdev = statistics.stdev(vp_qoe_list)
        vp_mean = statistics.mean(vp_qoe_list)
        service_qoe = vp_mean - vp_stdev

        request_vp_stdev = statistics.stdev(request_vp_qoe_list)
        request_vp_mean = statistics.mean(request_vp_qoe_list)
        request_qoe = request_vp_mean - request_vp_stdev

        service_qoe_ratio = service_qoe / request_qoe
        #print('service_qoe_ratio: ',service_qoe_ratio,' service qoe: ',service_qoe,' request qoe: ',request_qoe)
        service_qoe_ratio_sum += alpha*service_qoe_ratio - (1-alpha)*(backhaul_bw_ratio)
        vp_stdev_sum += vp_stdev
        ret_value=service_qoe_ratio_sum / num_request_per_seg
        if(ret_value>alpha):
            ret_value=alpha
    # aver_qoe=tot_qoe/num_tile_per_seg/num_request_per_seg
    # min_max_qoe_mean=min_max_diff_qoe_sum/num_request_per_seg
    # vp_stdev_mean=vp_stdev_sum/num_request_per_seg
    # aver_vp_qoe=viewport_qoe/num_viewport_tile-vp_stdev_mean
    return  ret_value # aver_vp_qoe





"""
backhaul_vers,select_vers --> [(seg_no, ver_list),...]
"""
def server_backhall_ver_decision(backhaul_vers,select_vers,backhaul_bw_gap,bitrate,qoe,num_tile_per_seg, num_ver_per_tile):
    num_req=len(backhaul_vers)
    #휴리스틱 값을 계산
    heuristic_value_list=[]
    num_ver_per_seg=num_tile_per_seg * num_ver_per_tile
    backhaul_bw_sum=0
    for req_idx in range(num_req):
        seg_no=backhaul_vers[req_idx][0]
        seg_start_in_ver_global = seg_no * num_ver_per_seg
        seg_end_in_ver_global = seg_start_in_ver_global + num_ver_per_seg
        heuristic_value_for_req=[]

        tmp_backhaul_vers=backhaul_vers[req_idx][1]
        tmp_select_vers=select_vers[req_idx][1]
        for tile_idx in range(num_tile_per_seg):
            if(tmp_backhaul_vers[tile_idx]!=-1):
                continue

            tile = tile_idx
            seg = seg_no
            select_ver = tmp_select_vers[tile_idx]
            select_ver_in_global = select_ver + seg_start_in_ver_global
            selected_qoe = qoe[select_ver_in_global]
            selected_bitrate = bitrate[select_ver_in_global]
            backhaul_bw_sum+=selected_bitrate
            tile_start_in_ver_seg = tile_idx * num_ver_per_tile
            tile_end_in_ver_seg = tile_start_in_ver_seg + num_ver_per_tile
            for ver_idx in range(tile_start_in_ver_seg,tile_end_in_ver_seg):
                ver = ver_idx
                if(ver_idx<select_ver):
                    continue

                veridx_in_global=ver_idx+seg_start_in_ver_global

                tmp_qoe=qoe[veridx_in_global]
                tmp_bitrate=bitrate[veridx_in_global]
                heuristic_value = (selected_qoe- tmp_qoe)/(selected_bitrate-tmp_bitrate)

                heuristic_value_for_req.append([seg,tile,ver,heuristic_value])
        if(len(heuristic_value_for_req)>0):
            heuristic_value_for_req.sort(
            key=lambda heuristic_value_for_req: heuristic_value_for_req[3])
            lowest_value_req=heuristic_value_for_req[0][3]
            heuristic_value_list.append([req_idx,lowest_value_req,heuristic_value_for_req])
    heuristic_value_list.sort(
            key=lambda heuristic_value_list: heuristic_value_list[1])
    #degradation 알고리즘 작동
    print(f"backhaul_bw_sum : {backhaul_bw_sum} backhaul_bw_gap: {backhaul_bw_gap} len_heuristic:{len(heuristic_value_list)}")
    while backhaul_bw_sum > backhaul_bw_gap and len(heuristic_value_list)>0:
        # Process the request with the lowest heuristic value

        req_idx, _, heuristic_values_for_req = heuristic_value_list[0]


        # print(
        #     f"len_heuristic:{len(heuristic_value_list)}")
        # Modify the version for the selected tile
        if len(heuristic_values_for_req)>0:
            seg_no, tile_idx, new_ver, _ = heuristic_values_for_req.pop(0)
            old_ver = select_vers[req_idx][1][tile_idx]
            if(new_ver<old_ver):
                if len(heuristic_values_for_req) == 0:
                    heuristic_value_list.pop(0)
                continue
            #print(new_ver)
            select_vers[req_idx][1][tile_idx] = new_ver
            seg_start_in_ver_global = seg_no * num_ver_per_seg
            # Update bandwidth
            old_bitrate = bitrate[old_ver+seg_start_in_ver_global]
            new_bitrate = bitrate[new_ver+seg_start_in_ver_global]
            backhaul_bw_sum += new_bitrate - old_bitrate

            # Recalculate heuristic values for the current request
            if len(heuristic_values_for_req)>0:
                for entry in heuristic_values_for_req:
                    _, _, ver_idx, _ = entry
                    selected_qoe = qoe[old_ver+seg_start_in_ver_global]
                    selected_bitrate = old_bitrate
                    tmp_qoe = qoe[ver_idx+seg_start_in_ver_global]
                    tmp_bitrate = bitrate[ver_idx+seg_start_in_ver_global]

                    if selected_bitrate - tmp_bitrate != 0:
                        entry[3] = (selected_qoe - tmp_qoe) / (selected_bitrate - tmp_bitrate)

                heuristic_values_for_req.sort(key=lambda x: x[3])  # Re-sort
                heuristic_value_list[0][1] = heuristic_values_for_req[0][3]
            else:
                # Remove the request if no more heuristic values
                heuristic_value_list.pop(0)

            heuristic_value_list.sort(key=lambda x: x[1])  # Re-sort the list
    # for req_idx in range(num_req):
    #     seg_no = backhaul_vers[req_idx][0]
    #     seg_start_in_ver_global = seg_no * num_ver_per_seg
    #     select_ver_in_global = select_ver + seg_start_in_ver_global
    return select_vers,backhaul_bw_sum

"""
backhaul_vers,select_vers --> [(ver_list),...]
"""
def backhall_ver_decision_for_train(backhaul_vers,select_vers,backhaul_bw_gap,bitrate,qoe,num_tile_per_seg, num_ver_per_tile):
    num_req=len(backhaul_vers)
    #휴리스틱 값을 계산
    heuristic_value_list=[]
    num_ver_per_seg=num_tile_per_seg * num_ver_per_tile
    backhaul_bw_sum=0
    for req_idx in range(num_req):
        seg_no=0
        seg_start_in_ver_global = seg_no * num_ver_per_seg
        seg_end_in_ver_global = seg_start_in_ver_global + num_ver_per_seg
        heuristic_value_for_req=[]

        tmp_backhaul_vers=backhaul_vers[req_idx]
        tmp_select_vers=select_vers[req_idx]
        for tile_idx in range(num_tile_per_seg):
            if(tmp_backhaul_vers[tile_idx]==-1):
                continue

            tile = tile_idx
            seg = seg_no
            select_ver = tmp_select_vers[tile_idx]

            select_ver_in_global = select_ver + seg_start_in_ver_global
            selected_qoe = qoe[select_ver_in_global]
            selectd_bitrate = bitrate[select_ver_in_global]
            backhaul_bw_sum+=selectd_bitrate
            tile_start_in_ver_seg = tile_idx * num_ver_per_tile
            tile_end_in_ver_seg = tile_start_in_ver_seg + num_ver_per_tile
            for ver_idx in range(tile_start_in_ver_seg,tile_end_in_ver_seg):
                ver = ver_idx
                if(ver_idx<select_ver):
                    continue

                veridx_in_global=ver_idx+seg_start_in_ver_global

                tmp_qoe=qoe[veridx_in_global]
                tmp_bitrate=bitrate[veridx_in_global]
                heuristic_value = (selected_qoe- tmp_qoe)/(selectd_bitrate-tmp_bitrate)

                heuristic_value_for_req.append([seg,tile,ver,heuristic_value])
        if(len(heuristic_value_for_req)==0):
            continue
        heuristic_value_for_req.sort(
            key=lambda heuristic_value_for_req: heuristic_value_for_req[3])
        lowest_value_req=heuristic_value_for_req[0][3]
        heuristic_value_list.append([req_idx,lowest_value_req,heuristic_value_for_req])
    heuristic_value_list.sort(
            key=lambda heuristic_value_list: heuristic_value_list[1])
    #degradation 알고리즘 작동

    while backhaul_bw_sum > backhaul_bw_gap and heuristic_value_list:
        # Process the request with the lowest heuristic value
        req_idx, _, heuristic_values_for_req = heuristic_value_list[0]

        # Modify the version for the selected tile
        if heuristic_values_for_req:
            seg_no, tile_idx, new_ver, _ = heuristic_values_for_req.pop(0)
            old_ver = select_vers[req_idx][tile_idx]
            select_vers[req_idx][tile_idx] = new_ver
            seg_start_in_ver_global = seg_no * num_ver_per_seg
            # Update bandwidth
            old_bitrate = bitrate[old_ver+seg_start_in_ver_global]
            new_bitrate = bitrate[new_ver+seg_start_in_ver_global]
            backhaul_bw_sum += new_bitrate - old_bitrate

            # Recalculate heuristic values for the current request
            if heuristic_values_for_req:
                for entry in heuristic_values_for_req:
                    _, _, ver_idx, _ = entry
                    selected_qoe = qoe[old_ver+seg_start_in_ver_global]
                    selected_bitrate = old_bitrate
                    tmp_qoe = qoe[ver_idx+seg_start_in_ver_global]
                    tmp_bitrate = bitrate[ver_idx+seg_start_in_ver_global]

                    if selected_bitrate - tmp_bitrate != 0:
                        entry[3] = (selected_qoe - tmp_qoe) / (selected_bitrate - tmp_bitrate)

                heuristic_values_for_req.sort(key=lambda x: x[3])  # Re-sort
                heuristic_value_list[0][1] = heuristic_values_for_req[0][3]
            else:
                # Remove the request if no more heuristic values
                heuristic_value_list.pop(0)

            heuristic_value_list.sort(key=lambda x: x[1])  # Re-sort the list

    return select_vers