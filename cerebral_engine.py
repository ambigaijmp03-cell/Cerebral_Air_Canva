import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import time

try:
    from mediapipe.python.solutions.face_mesh_connections import FACEMESH_TESSELATION
except ImportError:
    try:
        from mediapipe.solutions.face_mesh_connections import FACEMESH_TESSELATION
    except ImportError:
        FACEMESH_TESSELATION = frozenset([
            (0,9),(0,10),(0,17),(0,18),(0,37),(0,84),(0,181),(0,291),(0,314),(0,405),
            (1,44),(1,274),(2,11),(2,13),(2,302),(3,51),(4,5),(4,45),(4,51),(4,275),
            (4,281),(5,4),(5,45),(5,51),(5,195),(5,281),(6,168),(6,197),(6,419),
            (7,138),(7,177),(8,55),(8,168),(8,285),(9,0),(9,336),(10,0),(10,109),
            (10,151),(11,2),(11,302),(12,268),(13,2),(13,14),(14,13),(14,17),
            (17,0),(17,14),(17,84),(17,314),(18,0),(18,84),(18,314),(21,54),(21,162),
            (21,184),(22,23),(22,52),(22,53),(23,22),(23,52),(24,23),(24,53),(25,7),
            (25,112),(25,141),(26,22),(26,112),(26,465),(27,28),(28,27),(28,29),
            (29,28),(29,30),(30,29),(30,31),(31,30),(31,32),(32,31),(32,33),(33,32),
            (33,34),(34,33),(34,35),(35,34),(35,36),(36,35),(36,37),(37,0),(37,36),
            (37,72),(37,167),(37,203),(38,82),(38,86),(39,37),(39,40),(40,39),(40,185),
            (41,38),(41,81),(42,39),(42,74),(42,184),(43,42),(43,57),(44,1),(44,19),
            (44,45),(45,4),(45,44),(46,53),(46,63),(46,70),(47,114),(47,121),(47,126),
            (48,49),(49,48),(49,50),(50,49),(50,187),(51,3),(51,4),(51,5),(52,22),
            (52,23),(52,53),(53,24),(53,46),(53,52),(54,21),(54,103),(54,104),
            (55,8),(55,56),(55,107),(56,55),(57,43),(57,167),(58,57),(58,327),
            (59,166),(61,185),(61,146),(61,91),(62,76),(62,77),(63,46),(63,70),
            (64,62),(65,107),(66,69),(67,103),(68,104),(69,66),(69,104),(70,46),
            (70,63),(71,139),(72,37),(72,73),(73,72),(74,42),(74,184),(75,59),
            (76,62),(76,77),(77,76),(77,90),(78,95),(80,82),(81,41),(81,82),
            (82,38),(82,81),(83,1),(83,18),(84,17),(84,18),(85,86),(86,38),(86,85),
            (87,14),(87,178),(88,89),(89,88),(89,90),(90,77),(90,89),(91,61),
            (92,186),(93,234),(94,19),(95,78),(96,77),(97,98),(98,97),(99,100),
            (100,99),(101,50),(102,48),(103,54),(103,67),(104,54),(104,68),(104,69),
            (105,52),(105,66),(105,107),(106,55),(106,107),(107,55),(107,65),
            (107,105),(108,69),(108,151),(109,10),(109,108),(110,24),(110,25),
            (111,26),(111,117),(112,25),(112,26),(113,225),(114,47),(114,128),
            (115,131),(116,123),(117,111),(117,118),(118,117),(119,118),(120,119),
            (121,47),(121,128),(122,121),(122,244),(123,116),(123,147),(124,35),
            (125,7),(125,36),(126,47),(126,217),(127,34),(128,114),(128,121),
            (129,102),(130,33),(131,115),(132,130),(133,155),(133,243),(134,131),
            (135,169),(136,135),(137,136),(138,7),(138,172),(139,71),(140,135),
            (141,25),(141,246),(142,129),(143,111),(144,163),(145,144),(146,61),
            (147,123),(147,213),(148,177),(149,148),(150,136),(151,10),(151,108),
            (152,148),(153,22),(154,153),(155,133),(155,154),(156,70),(157,173),
            (158,155),(159,158),(160,159),(161,160),(162,21),(163,144),(164,0),
            (165,37),(166,59),(167,57),(168,6),(168,8),(169,135),(170,169),
            (171,140),(172,138),(173,157),(174,173),(175,171),(176,148),(177,7),
            (178,87),(179,86),(180,85),(181,0),(182,181),(183,42),(184,42),(184,74),
            (185,40),(185,61),(186,92),(187,50),(187,207),(188,183),(188,191),
            (189,190),(190,56),(191,188),(192,214),(193,168),(194,204),(195,5),
            (196,3),(197,6),(197,196),(198,197),(199,200),(200,199),(201,200),
            (202,210),(203,37),(203,142),(204,194),(205,36),(206,203),(207,187),
            (208,36),(209,208),(210,202),(211,210),(212,57),(213,147),(214,192),
            (215,138),(216,206),(217,126),(218,0),(219,48),(220,237),(221,34),
            (222,52),(223,157),(224,223),(225,113),(226,31),(227,226),(228,31),
            (229,228),(230,229),(231,230),(232,231),(233,232),(234,93),(235,234),
            (236,3),(237,220),(238,79),(239,238),(240,239),(241,240),(242,241),
            (243,133),(244,122),(245,244),(246,141),(247,30),(248,9),(249,330),
            (250,309),(251,389),(252,253),(253,254),(254,339),(255,339),(256,252),
            (257,258),(258,286),(259,257),(260,259),(261,260),(262,261),(263,466),
            (264,356),(265,353),(266,265),(267,269),(268,12),(269,267),(270,269),
            (271,265),(272,271),(273,275),(274,1),(274,19),(275,4),(275,274),
            (276,283),(277,350),(278,294),(279,278),(280,279),(281,4),(281,5),
            (282,281),(283,276),(284,283),(285,8),(285,56),(286,258),(287,290),
            (288,287),(289,288),(290,289),(291,0),(291,306),(292,291),(293,300),
            (294,278),(295,409),(296,293),(297,338),(298,301),(299,297),(300,293),
            (301,298),(302,2),(302,11),(303,302),(304,303),(305,304),(306,291),
            (307,306),(308,307),(309,250),(310,309),(311,310),(312,311),(313,312),
            (314,17),(314,18),(315,314),(316,315),(317,316),(318,317),(319,318),
            (320,319),(321,320),(322,321),(323,322),(324,323),(325,324),(326,325),
            (327,326),(328,327),(329,328),(330,329),(331,330),(332,331),(333,332),
            (334,333),(335,334),(336,9),(336,337),(337,336),(338,297),(338,337),
            (339,254),(339,255),(340,339),(341,340),(342,341),(343,342),(344,360),
            (345,346),(346,347),(347,348),(348,349),(349,350),(350,277),(350,349),
            (351,412),(352,345),(353,265),(354,19),(355,429),(356,264),(357,343),
            (358,357),(359,255),(360,344),(361,401),(362,398),(363,362),(364,363),
            (365,364),(366,365),(367,366),(368,367),(369,368),(370,369),(371,266),
            (372,340),(373,372),(374,373),(375,374),(376,375),(377,376),(378,395),
            (379,378),(380,379),(381,380),(382,381),(383,382),(384,383),(385,384),
            (386,385),(387,386),(388,387),(389,251),(390,389),(391,390),(392,391),
            (393,168),(394,393),(395,378),(396,347),(397,396),(398,362),(399,175),
            (400,377),(401,361),(402,168),(403,43),(404,403),(405,0),(406,405),
            (407,292),(408,407),(409,295),(410,287),(411,410),(412,351),(413,362),
            (414,413),(415,291),(416,415),(417,416),(418,262),(419,6),(420,360),
            (421,420),(422,335),(423,422),(424,335),(425,427),(426,425),(427,401),
            (428,262),(429,355),(430,394),(431,279),(432,431),(433,273),(434,369),
            (435,288),(436,434),(437,436),(438,309),(439,438),(440,36),(441,35),
            (442,441),(443,444),(444,443),(445,444),(446,445),(447,446),(448,447),
            (449,448),(450,449),(451,450),(452,451),(453,452),(454,453),(455,454),
            (456,455),(457,456),(458,309),(459,458),(460,459),(461,460),(462,461),
            (463,341),(464,463),(465,26),(466,263),(467,260),
        ])


# ─────────────────────────────────────────────
#  ANCHOR STATE
# ─────────────────────────────────────────────
class AnchorState:
    def __init__(self, x=200.0, y=200.0):
        self.x        = x
        self.y        = y
        self.target_x = x
        self.target_y = y
        # Use a smooth position history buffer to kill jitter
        self._hist_x  = np.full(6, x, dtype=np.float64)
        self._hist_y  = np.full(6, y, dtype=np.float64)
        self.locked   = False
        self.disintegration = 0.0

        n = 478
        self.p_vx = np.zeros(n, dtype=np.float32)
        self.p_vy = np.zeros(n, dtype=np.float32)
        self.p_ox = np.zeros(n, dtype=np.float32)
        self.p_oy = np.zeros(n, dtype=np.float32)

    def set_target(self, x, y):
        # Roll history buffer and take weighted mean — kills jitter completely
        self._hist_x = np.roll(self._hist_x, -1)
        self._hist_y = np.roll(self._hist_y, -1)
        self._hist_x[-1] = x
        self._hist_y[-1] = y
        # Weights: older samples matter less
        w = np.array([0.05, 0.08, 0.12, 0.18, 0.24, 0.33], dtype=np.float64)
        self.target_x = float(np.dot(self._hist_x, w))
        self.target_y = float(np.dot(self._hist_y, w))

    def update(self, lerp=0.18, drag=0.75, gravity=0.10):
        # Smooth lerp toward target — no velocity overshoot
        self.x += (self.target_x - self.x) * lerp
        self.y += (self.target_y - self.y) * lerp

        if self.disintegration > 0:
            scale = self.disintegration / 20.0
            self.p_vx *= drag
            self.p_vy *= drag
            self.p_vy += gravity
            self.p_ox += self.p_vx * scale
            self.p_oy += self.p_vy * scale
        elif self.locked:
            self.p_ox *= 0.78
            self.p_oy *= 0.78
            self.p_vx *= 0.40
            self.p_vy *= 0.40

    def kick_particles(self, drift=1.8):
        n = len(self.p_vx)
        self.p_vx[:] = np.random.uniform(-drift, drift, n).astype(np.float32)
        self.p_vy[:] = np.random.uniform(-drift, drift, n).astype(np.float32)

    def solidify(self, rate=5.0):
        self.locked = True
        self.disintegration = max(self.disintegration - rate, 0.0)

    def dissolve(self, rate=2.5, max_d=40.0):
        was_solid = self.disintegration < 1.0
        self.locked = False
        self.disintegration = min(self.disintegration + rate, max_d)
        return was_solid


# ─────────────────────────────────────────────
#  CEREBRAL ENGINE
# ─────────────────────────────────────────────
class CerebralEngine:
    def __init__(self, face_model='face_landmarker.task',
                       hand_model='hand_landmarker.task'):
        base_f = python.BaseOptions(model_asset_path=face_model)
        self.face_det = vision.FaceLandmarker.create_from_options(
            vision.FaceLandmarkerOptions(
                base_options=base_f,
                running_mode=vision.RunningMode.VIDEO,
                num_faces=1
            )
        )
        base_h = python.BaseOptions(model_asset_path=hand_model)
        self.hand_det = vision.HandLandmarker.create_from_options(
            vision.HandLandmarkerOptions(
                base_options=base_h,
                running_mode=vision.RunningMode.VIDEO,
                num_hands=1
            )
        )
        self.anchor = AnchorState()

    def detect(self, mp_image, timestamp_ms):
        f = self.face_det.detect_for_video(mp_image, timestamp_ms)
        h = self.hand_det.detect_for_video(mp_image, timestamp_ms)
        return f, h

    def process_hand(self, h_res, frame_w, frame_h):
        if not h_res.hand_landmarks:
            return False, 0, 0
        lms  = h_res.hand_landmarks[0]
        dist = np.hypot(lms[4].x - lms[8].x, lms[4].y - lms[8].y)
        ix   = lms[8].x * frame_w
        iy   = lms[8].y * frame_h
        if dist > 0.07:
            self.anchor.set_target(ix, iy)
            kicked = self.anchor.dissolve()
            if kicked:
                self.anchor.kick_particles()
        else:
            self.anchor.solidify()
        return True, int(ix), int(iy)

    def close(self):
        self.face_det.close()
        self.hand_det.close()


# ─────────────────────────────────────────────
#  RENDERER  — original depth-colored dot cloud
#  Camera feed visible in background
# ─────────────────────────────────────────────
class GhostRenderer:
    EDGE_THRESHOLD = 5.0

    @staticmethod
    def _depth_color(z):
        t = float(np.clip((z + 0.08) / 0.16, 0.0, 1.0))
        b = int(80  + t * 175)
        g = int(220 - t * 120)
        r = int(255 - t * 200)
        return (b, g, r)

    @staticmethod
    def _dot_radius(z):
        return max(1, int(2.5 - z * 18))

    def draw_ghost(self, frame, face_res, anchor):
        if not face_res.face_landmarks:
            return
        h, w = frame.shape[:2]
        face_lms = face_res.face_landmarks[0]
        nose = face_lms[1]
        nw, nh = nose.x * w, nose.y * h

        pts = []
        for i, lm in enumerate(face_lms):
            rel_x = (lm.x * w) - nw
            rel_y = (lm.y * h) - nh
            proj  = 1.0 + lm.z * 0.35
            cx    = int(anchor.x + rel_x * proj + anchor.p_ox[i])
            cy    = int(anchor.y + rel_y * proj + anchor.p_oy[i])
            color  = self._depth_color(lm.z)
            radius = self._dot_radius(lm.z)
            alpha  = float(np.clip(1.0 - abs(lm.z) * 6, 0.3, 1.0))
            pts.append((lm.z, cx, cy, color, radius, alpha, i))

        pts.sort(key=lambda p: p[0], reverse=True)

        if anchor.disintegration < self.EDGE_THRESHOLD:
            ea = 1.0 - anchor.disintegration / self.EDGE_THRESHOLD
            idx_map = {p[6]: p for p in pts}
            for (a_i, b_i) in FACEMESH_TESSELATION:
                if a_i not in idx_map or b_i not in idx_map:
                    continue
                _, ax, ay, ac, _, _, _ = idx_map[a_i]
                _, bx, by, bc, _, _, _ = idx_map[b_i]
                if not (0 < ax < w and 0 < ay < h and 0 < bx < w and 0 < by < h):
                    continue
                ec = tuple(int((ac[k]+bc[k])/2 * ea * 0.55) for k in range(3))
                cv2.line(frame, (ax, ay), (bx, by), ec, 1, cv2.LINE_AA)

        for _, cx, cy, color, radius, alpha, _ in pts:
            if 0 < cx < w and 0 < cy < h:
                c = tuple(int(v * alpha) for v in color)
                cv2.circle(frame, (cx, cy), radius, c, -1, cv2.LINE_AA)

    @staticmethod
    def draw_ui(frame, anchor, hand_pos=None):
        h, w = frame.shape[:2]
        d_pct = int(anchor.disintegration / 40.0 * 100)
        if anchor.locked:
            status = "SOLID"
            col    = (180, 255, 180)
            cv2.drawMarker(frame, (int(anchor.x), int(anchor.y)),
                           (255, 255, 255), cv2.MARKER_CROSS, 18, 1, cv2.LINE_AA)
        else:
            status = f"DISSOLVING  {d_pct}%"
            col    = (80, 220, 255)
            if hand_pos:
                cv2.circle(frame, hand_pos, 10, (0, 220, 220), 2, cv2.LINE_AA)
        cv2.putText(frame, status,
                    (20, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.65, col, 2, cv2.LINE_AA)
        cv2.putText(frame, "PINCH = lock    OPEN = float",
                    (20, h - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.48,
                    (160, 160, 160), 1, cv2.LINE_AA)


# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────
def main():
    engine   = CerebralEngine()
    renderer = GhostRenderer()
    cap      = cv2.VideoCapture(0)

    try:
        while cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)
            h, w  = frame.shape[:2]
            ts    = int(time.time() * 1000)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB,
                              data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            f_res, h_res = engine.detect(mp_img, ts)
            has_hand, ix, iy = engine.process_hand(h_res, w, h)
            engine.anchor.update()
            renderer.draw_ghost(frame, f_res, engine.anchor)
            renderer.draw_ui(frame, engine.anchor,
                             hand_pos=(ix, iy) if has_hand else None)
            cv2.imshow("Cerebral Anchor", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        engine.close()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()