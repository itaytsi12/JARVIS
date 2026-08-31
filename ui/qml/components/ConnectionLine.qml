import QtQuick

// A thin glowing line from the central hub to one model node.
//
// Drawn as a rotated Rectangle rather than a Canvas path: it is a single
// batched quad, so twenty of these cost essentially nothing, and the
// "flowing data" highlight is a transform animation the render thread
// handles without ever repainting anything.
Item {
    id: line

    // Endpoints, in this item's parent coordinates.
    property real x1: 0
    property real y1: 0
    property real x2: 0
    property real y2: 0

    property color accent: "#2f8fd8"
    property real intensity: 0.35
    // Set while the node at the far end is thinking/active: the line
    // brightens and a light point travels along it.
    property bool flowing: false

    readonly property real dx: x2 - x1
    readonly property real dy: y2 - y1
    readonly property real length: Math.sqrt(dx * dx + dy * dy)
    readonly property real angle: Math.atan2(dy, dx) * 180 / Math.PI

    x: x1
    y: y1
    width: length
    height: 1
    transformOrigin: Item.Left
    rotation: angle

    Behavior on intensity { NumberAnimation { duration: 300 } }

    // The line itself.
    Rectangle {
        anchors.fill: parent
        color: line.accent
        opacity: line.intensity
    }

    // A wider, much fainter band underneath, so the line reads as glowing
    // rather than as a hairline.
    Rectangle {
        anchors.centerIn: parent
        width: parent.width
        height: 3
        color: line.accent
        opacity: line.intensity * 0.16
    }

    // The travelling light point. Only exists while the far node is busy.
    Rectangle {
        id: pulse
        width: 4
        height: 4
        radius: 2
        y: -1.5
        color: line.accent
        visible: line.flowing
        opacity: 0.9
        NumberAnimation on x {
            running: pulse.visible
            loops: Animation.Infinite
            from: 0
            to: line.length
            duration: 1500
            easing.type: Easing.InOutSine
        }
    }
}
