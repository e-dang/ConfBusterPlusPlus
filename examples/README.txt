The following pymol commands were used to produce the effects seen in the figures.

split_states all
hide (hydro)
util.cbaw all
set stick_transparency, 0.65
set stick_radius, 0.05
select <name of lowest energy conformer>
set stick_transparency, 0, sele
set stick_radius, 0.25, sele
util.cbag sele