# The seat's skill container (the skill in the loop: sport.toml [modes.low.inloop], skilleval/inloop.py): the
# simulator's image without the simulator, the packages the seat promises a skill ([modes.low] eval_pip), the
# low level's toolkit at the seat's paths and skilleval's host, which runs the player's skill as nobody.
FROM ghcr.io/mll-lab-nu/eai-olympiad:0.1.0
RUN pip install --no-cache-dir scipy==1.18.1
COPY skilleval /opt/skilleval
COPY files/opt/robot/assets /opt/robot/assets
COPY files/opt/robot/hit_cases_public.jsonl /opt/robot/hit_cases_public.jsonl
COPY files/opt/robot/lib/tt_kinematics.py /opt/robot/lib/tt_kinematics.py
COPY files/opt/robot/lib/tt_servo.py /opt/robot/lib/tt_servo.py
COPY files/opt/robot/robot_only.xml /opt/robot/robot_only.xml
COPY files/opt/robot/toolkit.toml /opt/robot/toolkit.toml
WORKDIR /tmp
