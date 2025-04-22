def show_info_data(col, label, n_points_valides, n_points_total):
    return col.markdown(f"""
            **{label}**  
            {n_points_valides} / {n_points_total}  
            Tx couverture : {(n_points_valides / n_points_total * 100):.1f}%
            """)

def show_info_metric(col, label, metric):
    return col.markdown(f"""
            **{label}**  
            {metric:.3f}
            """)