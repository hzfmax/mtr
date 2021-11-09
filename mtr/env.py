import numba as nb
import numpy as np
from gym import Env, spaces
from utils.loader import get_EAL


# coefficients for cost function
ctime = 6.945 / 3600  # pounds per second
cost_per_dist = 3.046e-4  # pounds per meter

cbrdt = 6.46 / 3600  # pounds per second
cwait = 2.5 * cbrdt  # pounds per second
ceng = 0.1077 / 3600  # pounds
beta1 = 16.6
beta2 = 0.366
beta3 = 0.0260

@nb.njit(
    "Tuple((i8[:], f8[:], f8, f8[:]))(i8[:], i8[:], i8[:], i8, i8, f8[:, :, :], f8[:, :])",
    fastmath=False,
    cache=True)
def board(eff_board_t, dpt, offset, T_in, cap, TDD, CDD):
    dpt_offset = dpt - offset  # actual offset departure time
    board_t_max = np.minimum(dpt_offset, T_in)  # bounday of maximum boarding time

    num_plat = len(dpt)  # number of platforms
    num_stat = np.int64(num_plat / 2) - 1  # number of stations

    # update the boarding time at the terminal platform
    eff_board_t[-1] = board_t_max[-1]
    eff_board_t[num_stat] = board_t_max[num_stat]

    # initialize
    pwt = 0.  # total passenger waiting time
    psg_brd_route = np.zeros(num_plat - 2, dtype=np.float64)  # number of passenger on the route
    board_psg_to = np.zeros(num_plat, dtype=np.float64)  # number of passenger onboard
    psg_left = np.zeros(num_plat, dtype=np.float64)  # number of left passengers

    # main loop
    for s in range(num_plat - 1):
        board_psg_to[s] = 0.  # Alighting first
        if s != num_stat:
            # Boarding
            room = cap - board_psg_to.sum()  # left room on the train
            if eff_board_t[s] < board_t_max[s]:
                dwait = CDD[eff_board_t[s]:board_t_max[s], s]
                if eff_board_t[s] > 0:
                    dwait = dwait - CDD[eff_board_t[s] - 1, s]

                # calculate the local effective boarding time
                t_idx = np.searchsorted(dwait, room, 'left')
                board_t = t_idx + eff_board_t[s]

                # boarding process
                psg_to_board = TDD[eff_board_t[s]:board_t, s, :]  # (time, number)
                board_psg_to = board_psg_to + psg_to_board.sum(0)

                # Calculate passenger waiting time
                # Definition: interval between the Arrivals of Passengers and Trains
                waitn = psg_to_board.sum(1)  # number of waiting passengers

                # related waiting times with time calibration
                waitt = dpt_offset[s] - 0.5 - np.arange(
                    eff_board_t[s], board_t, dtype=np.float64)

                pwt += np.dot(waitt, waitn)

                # train load for each route
                psg_brd_route[s if s < num_stat else s - 1] = board_psg_to.sum()

                # number of left passengers at the platform
                psg_left[s] = dwait[-1] - dwait[t_idx - 1]

                # update the effective boarding time
                eff_board_t[s] = board_t

    return eff_board_t, psg_left, pwt, psg_brd_route


@nb.njit("Tuple((f8, f8))(i8[:], i8[:], f8[:], f8[:], f8, f8, f8, f8, f8)",
         fastmath=True,
         cache=True)
def cost_func(runt, dwl, dis, psg_brd_route, pwt, alpha, ofix, Wtrain, Wman):
    v = dis / runt
    eng = ((beta1 + beta2 * v + beta3 * v**2) *
           (Wtrain + Wman * psg_brd_route) * dis).sum() * ceng
    pwc = alpha * cwait * pwt
    opc = (1 - alpha) * (eng + ofix + ctime * runt.sum())
    return pwc, opc


@nb.njit(fastmath=True)
def maximum_filter(x):
    x = np.maximum(x, 0)
    max_x = 0
    for i in range(len(x)):
        if x[i] > max_x:
            max_x, x[i] = x[i], x[i] - max_x
        else:
            x[i] = 0
    return x


@nb.njit('i8[:](i8[:],i8[:],i8[:],i8,i8,i8,i8[:],i8[:],i8[:],i8,i8[:])', fastmath=True, cache=True)
def xvalid(act, arv, dpt, hdw_sft, stock, num_plat, var_max, var_int, var_min, train, ftrain):
    # compaalgoheadway, short-term rs headway
    hdw = max(hdw_sft + dpt[0] - arv[0], ftrain[train] - arv[0])

    # Long term rolling stock index
    if train % stock != 0:
        idx = train + stock - (train % stock)  # index of previous trip
        hdw = max(hdw, ftrain[idx] - var_max[0] * (idx - train) - arv[0])

    if act[0] < hdw:
        act[0] = np.ceil((hdw - var_min[0]) / var_int[0]) * var_int[0] + var_min[0]

    # Safety headway, unsafe when > 0, refer to equation (26)
    safe = dpt + hdw_sft - arv[0] - act.cumsum()[0::2]

    if safe.max() > 0:
        safe = maximum_filter(safe)  # adjust those greater than 0

        # modify running times and dwell times
        sps = var_max - act  # margin to modify
        for i in range(num_plat - 1, -1, -1):
            if safe[i] > 0:
                pid = i * 2  # first adjust the dwell time
                if sps[pid] >= safe[i]:
                    act[pid] += np.int64(np.ceil(
                        safe[i] / var_int[pid])) * var_int[pid]
                elif i != 0:
                    safe[i] -= sps[pid]
                    act[pid] = var_max[pid]
                    if sps[pid - 1] >= safe[i]:
                        act[pid -
                            1] += np.int64(np.ceil(
                                safe[i] / var_int[pid - 1])) * var_int[pid - 1]
                    else:
                        safe[i - 1] += safe[i] + act[pid - 1] - var_max[pid - 1]
                        act[pid - 1] = var_max[pid - 1]

    # Maximum headway constraints between by-pass stations
    mxhdw = arv[0] + act.cumsum()[0::2] - var_max[0] - arv
    if mxhdw.max() > 0:
        mxhdw = maximum_filter(mxhdw)  # adjust those greater than 0

        sps = act - var_min  # the decision space to modify
        for idx in range(1, num_plat):
            if mxhdw[idx] > 0:
                pid = idx * 2
                if sps[pid] >= mxhdw[idx]:
                    act[pid] -= np.int64(np.ceil(mxhdw[idx] / var_int[pid])) * var_int[pid]
                else:
                    act[pid] = var_min[pid]
                    act[pid - 1] -= np.int64(
                        np.ceil((mxhdw[idx] - sps[pid]) /
                                var_int[pid - 1])) * var_int[pid - 1]

    return act


@nb.njit(fastmath=True)
def ODshape(tvd):
    shp = tvd.shape
    output = np.zeros((shp[0], shp[1] * 2, shp[2] * 2), dtype=np.float64)
    output[:, :shp[1], :shp[2]] = np.triu(tvd, 1)
    output[:, shp[1]:, shp[2]:] = np.tril(tvd, -1)[:, ::-1, ::-1]
    return output


class TubeEnv(Env):
    name = 'LineEnv'
    version = 0

    def __init__(
            self,
            data,
            alpha=0.5,
            stochastic=False,
            random_factor=0.25,
            headway_opt=7,
            headway_int=30,
            dwell_int=5,
            dwell_opt=5,
            run_int=10,
            run_opt=5,
            seed=None
    ):
        super().__init__()
        # Time config
        self.t_start = int(data['t_start'])
        self.t_end = int(data['t_end'])
        self.T_in = np.int64((self.t_end - self.t_start) * 3600)  # Total time intervals
        self.alpha = alpha  # weight coefficient
        self.capacity = data['capacity']  # vehicle capacity
        self.turn = data['turnaround']  # U-turn time for LOWO and HUHI
        self.stock_size = data['stock_size'] # stock size
        self.stochastic = stochastic  # Wether this is a stochastic env
        self.headway_safe = data['sft_hdw']  # safety headway
        self.max_svs = np.int64((self.t_end - self.t_start) * 3600 / data['dft_hdw']) + 1
        self.random_factor = random_factor
        self.Wtrain = data['train_weight']
        self.Wman = data['psg_weight']

        # Platform information
        self.num_stat = len(data['routes'])
        self.num_plat = self.num_stat * 2
        self.routes = list(
            map(lambda x: x + "O", data['routes'][:self.num_stat])) + list(
                map(lambda x: x + "I", data['routes'][:self.num_stat][::-1]))

        # Section running time & distance
        self.run = np.hstack(
            ([data['dft_hdw']], data['run'][:self.num_stat - 1], [self.turn[0]],
             data['run'][-self.num_stat + 1:])).astype(np.float64)

        self.distance = np.hstack(
            (data['distance'][:self.num_stat - 1],
             data['distance'][-self.num_stat + 1:])).astype(np.float64)

        self.opr_fix = self.distance.sum() * cost_per_dist

        # OD matrix
        d = np.moveaxis(data['demand'], 2, 0)
        self.tvd = d[(self.t_start - 5) * 60:(self.t_end - 5) *
                     60, :int(self.num_stat), :][..., :int(self.num_stat)]
        self.TDD = np.repeat(ODshape(self.tvd) / 60, 60, axis=0)  # to be used for static env

        # Range of variables
        self.nopt = np.array([[headway_opt] + [run_opt] *
                              (self.num_stat - 1) + [1] + [run_opt] *
                              (self.num_stat - 1),
                              ([dwell_opt] * (self.num_stat - 1) + [1]) * 2],
                             dtype=np.int64).flatten('F') - 1

        self.var = np.array([self.run, data['dwell']],
                            dtype=np.int64).flatten('F')
        self.var_int = np.array(
            [[headway_int] + [run_int] * (self.num_plat - 1),
             [dwell_int if x else 0 for x in data['dwell']]],
            np.int64).flatten('F')

        self.var_max = (self.var + self.nopt / 2 * self.var_int).astype(np.int64)
        self.var_min = (self.var - self.nopt / 2 * self.var_int).astype(np.int64)
        print(self.var, self.var_min, self.var_max)

        # Init state
        self.offset = np.append(0, self.var_min[1:]).cumsum()[0::2].astype(np.int64)
        self.istate = np.hstack([[0],
                                 np.append(0, self.var_min[1:]).cumsum(),
                                 np.zeros(self.num_plat)]).astype(np.float64)

        # RL space
        self.action_space = spaces.Box(low=-1,
                                       high=1,
                                       shape=(2 * self.num_plat, ), dtype=np.float64)
        self.observation_space = spaces.Box(
            np.zeros(self.num_plat * 3 + 1),
            np.concatenate([[self.max_svs / 2],
                            np.full(2 * self.num_plat, self.T_in),
                            np.full(self.num_plat, 500)], 0),
            dtype=np.float64)

        self.u_indice = list(range(self.num_stat - 1)) + list(
            range(self.num_plat - self.num_stat, self.num_plat - 1))

        self.seed(seed)
        assert self.TDD.shape[0] == self.T_in

    def reset(self):
        # State: timetable and left passenger for previous train
        self.state = self.istate.copy()
        self.train = 0
        self.feasible_train = [0] * self.stock_size
        self.arv = np.zeros(self.num_plat, np.int64)
        self.eff_board_t = np.zeros(self.num_plat, np.int64)

        if self.stochastic:
            # This is faster than calcuation based on repeated matrix
            self.TDD = self.tvd * np.maximum(
                np.random.normal(1, self.random_factor, self.tvd.shape), 0.)
            self.TDD = np.repeat(self.TDD / 60, 60, axis=0)
            self.TDD = ODshape(self.TDD)

        self.CDD = self.TDD.cumsum(0).sum(2)
        return self.state, False

    def step(self, action):
        # translate the action
        act = (np.around(self.nopt * (action + 1) / 2) * self.var_int +
               self.var_min).astype(np.int64)

        # validation
        if self.train > 0:
            act = xvalid(act, self.arv, self.dpt, self.headway_safe,
                         self.stock_size, self.num_plat, self.var_max,
                         self.var_int, self.var_min, self.train,
                         np.array(self.feasible_train, dtype=np.int64))
        act1=act
        # generate new timetable
        act[0] = act[0] + self.arv[0]  # arrive at the starting station
        ttb = act.cumsum(dtype=np.int64)

        # update arrival times for all nodes
        self.arv = ttb[0::2]

        # update departure times for all nodes
        self.dpt = ttb[1::2]

        # log the feasible time for this train for a potential next trip
        self.feasible_train.append(self.dpt[-1] + self.turn[-1])

        # boarding process of the service at all nodes
        self.eff_board_t, psg_left, pwt, psg_brd_route = board(
            self.eff_board_t, self.dpt, self.offset, self.T_in, self.capacity,
            self.TDD, self.CDD)

        # Weather all passengers have been served
        done = np.all(self.eff_board_t[:-1] == self.T_in)

        # update train index
        self.train += 1

        # calculate related costs for operating the service
        pwc, opc = cost_func(act[2::2][self.u_indice],
                             act[1::2][self.u_indice], self.distance,
                             psg_brd_route, pwt, self.alpha, self.opr_fix,
                             self.Wtrain, self.Wman)

        # get the state representation
        self.state = np.concatenate([[self.train], ttb, psg_left], 0)

        return act1, self.state, pwc, opc, done

    def seed(self, seed=None):
        self.np_random = np.random.RandomState()
        if np.isscalar(seed):
            seed = int(seed * np.pi)
            self._seed = seed
            self.np_random.seed(self._seed)
            self.action_space.seed(self._seed)
        else:
            self._seed = None
        return [self._seed]

    def __str__(self):
        return  f"{self.name}-v{self.version} "


if __name__ == "__main__":

    from utils.loader import get_EAL
    data = get_EAL(read=False, store=True)
    data['demand'] = data['demand'] * 0.25
    env = TubeEnv(data)
    # print(data)

    obs = env.reset()
    print("totol number of passengers:", int(env.CDD[-1, ...].sum()))
    done = False

    # no optimization
    topc, tpwc = 0, 0
    timetable = []
    while not done:
        act = np.zeros(env.action_space.shape)
        actt, obs2, pwc, opc, done = env.step(act)
        timetable.append(actt)
        topc += opc
        tpwc += pwc
    print(np.array(timetable))
    print(topc, tpwc, topc + tpwc, env.train, "\n", f"Last departure: {int(env.dpt[0])}", '\n', env.eff_board_t)
    print("completed")


    # acta = []
    # tpwc = []
    # topc = []
    # cost=[]
    # for z in range(2):
    #     act = np.zeros(48)
    #     act[0] = act[0] +z*headways
    #     for i in range(23):
    #         act[i*2+2]=runtime_1[i]
    #     for i in range(24):
    #         act[i*2+1]= dwelltime[i]
    #     act[0] = act[0] + arrivetime[0]


    #     ttb = act.cumsum(dtype=np.int64)
    #     arv = ttb[0::2]
    #     ttb = act[1::2].cumsum(dtype=np.int64)

    #     env.eff_board_t, psg_left, pwt, psg_brd_route = board(env.eff_board_t, ttb, env.offset, env.T_in, env.capacity, env.TDD, env.CDD)
    #     eng = ((beta1 + beta2 * v + beta3 * v**2) *
    #         (W_train + W_man * psg_brd_route) * distance).sum() * ceng
    #     pwc = env.alpha * cwait * pwt
    #     opc = (1 - env.alpha) * (eng + env.opr_fix + ctime * np.array(runtime).sum())
    #     tpwc.append(pwc)
    #     topc.append(opc)
    #     cost.append(pwc+opc)
    #     acta.append(act)
    # print('tpwc=',tpwc)
    # print('topc =',topc )
    # print('cost =',cost )
    # print('act =',acta)
