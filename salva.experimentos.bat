set SUBDIRS=251105_ir_study_base_p 251105_ir_study_base_p_asset_i 251105_ir_study_base_p_asset_i_canary 251105_ir_study_base_p_asset_i1 251105_ir_study_base_p_asset_j 251105_ir_study_base_p_asset_j1 251105_ir_study_base_p_c 251105_ir_study_base_p_c1 251105_ir_study_base_p_p 251105_ir_study_base_p_p1 251105_ir_study_base_psi 251105_ir_study_base_psi_asset_i 251105_ir_study_base_psi_asset_i_canary 251105_ir_study_base_psi_asset_i1 251105_ir_study_base_psi_asset_j 251105_ir_study_base_psi_asset_j1 251105_ir_study_base_psi_c 251105_ir_study_base_psi_c1 251105_ir_study_base_psi_p 251105_ir_study_base_psi_p1

for %%d in (%SUBDIRS%) do (
    if exist "c:\experiments\%%d\results.txt" (
        copy "c:\experiments\%%d\results.gdt" "C:\experiments\backups\%%d.results.gdt"
        for %%z in (prob_bankruptcy rationing bankruptcies asset_i asset_j bad_debt c equity interest_rate) do (
            copy "c:\experiments\%%d\%%z.png" "C:\experiments\backups\%%d.%%z.png"
            copy "c:\experiments\%%d\%%z.txt" "C:\experiments\backups\%%d.%%z.txt"
        )
    )
)