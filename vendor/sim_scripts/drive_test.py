#!/usr/bin/env python3
"""Proof that the simulator accepts /cmd_vel: drive forward ~4 s and measure
how far /state_estimation moved.

The official stack ships the simulator only -- there is no planner -- so
"exploration" means we publish the twist ourselves.  This is the smoke test
that the door is actually open.
"""
import math

import rospy
from geometry_msgs.msg import TwistStamped
from nav_msgs.msg import Odometry

DUR = 4.0
SPEED = 1.0          # m/s
YAW = 0.35           # rad/s, gentle left turn so we also see heading change


def pose(msg):
    p = msg.pose.pose.position
    q = msg.pose.pose.orientation
    yaw = math.atan2(2 * (q.w * q.z + q.x * q.y),
                     1 - 2 * (q.y * q.y + q.z * q.z))
    return p, yaw


def main():
    rospy.init_node("drive_test", anonymous=True, disable_signals=True)
    pub = rospy.Publisher("/cmd_vel", TwistStamped, queue_size=1)

    rospy.loginfo("waiting for /state_estimation ...")
    od0 = rospy.wait_for_message("/state_estimation", Odometry, timeout=30.0)
    p0, y0 = pose(od0)
    t0 = rospy.Time.now()
    print("t0 = %.3f s   start (%.3f, %.3f, %.3f)  yaw %.2f deg"
          % (t0.to_sec(), p0.x, p0.y, p0.z, math.degrees(y0)))

    rate = rospy.Rate(20)
    n = 0
    while not rospy.is_shutdown() and (rospy.Time.now() - t0).to_sec() < DUR:
        m = TwistStamped()
        m.header.stamp = rospy.Time.now()
        m.twist.linear.x = SPEED
        m.twist.angular.z = YAW
        pub.publish(m)
        n += 1
        rate.sleep()

    for _ in range(10):                       # explicit stop
        m = TwistStamped()
        m.header.stamp = rospy.Time.now()
        pub.publish(m)
        rospy.sleep(0.05)

    od1 = rospy.wait_for_message("/state_estimation", Odometry, timeout=30.0)
    p1, y1 = pose(od1)
    d = math.hypot(p1.x - p0.x, p1.y - p0.y)
    dt = (rospy.Time.now() - t0).to_sec()
    print("t1 = %.3f s   end   (%.3f, %.3f, %.3f)  yaw %.2f deg"
          % (rospy.Time.now().to_sec(), p1.x, p1.y, p1.z, math.degrees(y1)))
    print("published %d twists over %.2f s" % (n, dt))
    print("MOVED %.3f m   dyaw %.2f deg   (mean speed %.2f m/s)"
          % (d, math.degrees(y1 - y0), d / max(dt, 1e-6)))
    print("VERDICT:", "PASS - the car responds to /cmd_vel" if d > 0.5
          else "FAIL - no motion, /cmd_vel is not being honoured")


if __name__ == "__main__":
    main()
