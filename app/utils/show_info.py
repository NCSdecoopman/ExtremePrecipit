def show_info_data(col, label, n_points_valides, n_points_total):
    both_defined = n_points_valides is not None and n_points_total is not None
    if both_defined:
        return col.markdown(f"""
                **{label}**  
                {n_points_valides} / {n_points_total}  
                Tx couverture : {(n_points_valides / n_points_total * 100):.1f}%
                """)
    else:
        return None

def show_info_metric(col, label, metric):
    if metric is not None:
        return col.markdown(f"""
                **{label}**  
                {metric:.3f}
                """)
    else:
        return None