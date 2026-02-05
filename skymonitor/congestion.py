import argparse

import numpy as np

from einops import rearrange

from skymonitor.simbarca_explore import initialize_dataset

SPEED_LIMIT = 50/3.6 # 50 km/h given in unit m/s because of SI units used in flow and density

def get_congestion_score(
    flow: np.ndarray,
    density: np.ndarray,
    density_threshold: float = 1e-3,
    ff_quantile: float = 1.0,
) -> np.ndarray:

    assert flow.shape == density.shape, "Flow and density arrays must have the same shape."

    S, T, P = flow.shape
    flow, density = rearrange(flow, 'S T P -> P (S T)'), rearrange(density, 'S T P -> P (S T)')

    # compute speed only where density is valid to avoid divide-by-zero warnings
    speed = np.full(flow.shape, np.nan, dtype=float)
    np.divide(flow, density, out=speed, where=density > density_threshold)
    
    # free flow speed of each road segment (ff_quantile observed speed)
    # 1.0 means maximum speed 
    ff_speed = np.full(speed.shape[0], SPEED_LIMIT, dtype=float)
    non_empty = np.isfinite(speed).any(axis=1)
    ff_speed[non_empty] = np.nanquantile(speed[non_empty], ff_quantile, axis=1)

    speed = np.nan_to_num(speed, nan=SPEED_LIMIT) # we have nans in speed because the density is zero
    congestion_score = np.clip(1 - (speed / ff_speed[:, None]), 0.0, 1.0)

    congestion_score = rearrange(congestion_score, 'P (S T) -> S T P', S=S)

    return congestion_score


def get_congestion_change(congestion_score: np.ndarray, entry_thd: float, exit_thd: float) -> np.ndarray:
    """Compute congestion state using a 2-step lookahead and hysteresis thresholds.

    Args:
        congestion_score: Array with shape (S, T, P) representing congestion scores.
        entry_thd: Congestion entry threshold (between 0 and 1).
        exit_thd: Congestion exit threshold (between 0 and 1).
    """
    assert congestion_score.ndim == 3, "Expected congestion_score with shape (S, T, P)."
    S, T, P = congestion_score.shape
    state, enters, exits = [np.zeros((S, T, P), dtype=bool) for _ in range(3)]
    current = np.zeros((S, P), dtype=bool)

    for t in range(T):

        # the last two time steps will be kept the same as previous time step
        if t + 2 < T:
            # a segment ENTERS congestion if it is not currently congested and then becomes congested for the next two time steps
            enter_congestion = (
                (~current)
                & (congestion_score[:, t + 1, :] > entry_thd)
                & (congestion_score[:, t + 2, :] > entry_thd)
            )
            # a segment EXITS congestion if it is currently congested and then becomes uncongested for the next two time steps
            # this criterion is stricter than the entering because we want to avoid rapid toggling
            exit_congestion = (
                current
                & (congestion_score[:, t + 1, :] < exit_thd)
                & (congestion_score[:, t + 2, :] < exit_thd)
            )
            # update the congestion state. 
            # a segment is congested if it was previously uncongested and enters congestion (first part), 
            # if it exits congestion it becomes uncongested (second part)
            current = (current | enter_congestion) & ~exit_congestion
            enters[:, t, :] = enter_congestion
            exits[:, t, :] = exit_congestion

        state[:, t, :] = current

    return state, enters, exits

if __name__ == "__main__":

    from skymonitor.visualize import visualize_map_scores

    parser = argparse.ArgumentParser(description='Visualize congestion scores and state.')
    parser.add_argument('--split', type=str, default='test', choices=['train', 'test'], help='Dataset split to visualize.')
    parser.add_argument('--density-threshold', type=float, default=1e-3, help='Minimum density to compute speed; lower values treated as empty.')
    parser.add_argument('--ff-quantile', type=float, default=0.85, help='Quantile used to estimate free-flow speed per segment. 1.0 means maximum observed speed.')
    parser.add_argument('--entry-thd', type=float, default=0.5, help='Congestion entry threshold. Value above is considered congested.')
    parser.add_argument('--exit-thd', type=float, default=0.7, help='Congestion exit threshold. Value below is not congested.')
    args = parser.parse_args()

    trainset, test_set = initialize_dataset()
    vis_set = trainset if args.split == 'train' else test_set

    flow, density = vis_set.veh_flow_3min, vis_set.veh_density_3min
    congestion_score = get_congestion_score(flow, density, density_threshold=args.density_threshold, ff_quantile=args.ff_quantile)
    congestion_state, enters, exits = get_congestion_change(congestion_score, entry_thd=args.entry_thd, exit_thd=args.exit_thd)

    visualize_map_scores(
        congestion_score,
        node_coordinates=vis_set.node_coordinates,
        info={
            'session_ids': vis_set.session_ids,
            'demand_scales': vis_set.demand_scales,
        },
        note='congestion_score_{}_ffq{}_en{}_ex{}'.format(args.split, args.ff_quantile, args.entry_thd, args.exit_thd),
    )
    
    visualize_map_scores(
        congestion_state.astype(float),
        node_coordinates=vis_set.node_coordinates,
        info={
            'session_ids': vis_set.session_ids,
            'demand_scales': vis_set.demand_scales,
        },
        note='congestion_state_{}_ffq{}_en{}_ex{}'.format(args.split, args.ff_quantile, args.entry_thd, args.exit_thd),
    )

    visualize_map_scores(
        np.logical_or(enters, exits).astype(float),
        node_coordinates=vis_set.node_coordinates,
        info={
            'session_ids': vis_set.session_ids,
            'demand_scales': vis_set.demand_scales,
        },
        note='congestion_change_{}_ffq{}_en{}_ex{}'.format(args.split, args.ff_quantile, args.entry_thd, args.exit_thd),
    )
