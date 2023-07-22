use phobos::{QueueRequest, QueueType};
use phobos::domain::{Compute, Graphics, Transfer};

mod framework;

#[test]
pub fn can_initialize() {
    let _context = framework::make_context().expect("Initialization should work");
}

#[test]
pub fn multi_queue_initialize() {
    let context = framework::make_context_with_queues([
        QueueRequest {
            dedicated: false,
            queue_type: QueueType::Graphics,
        },
        QueueRequest {
            dedicated: true,
            queue_type: QueueType::Transfer,
        },
        QueueRequest {
            dedicated: true,
            queue_type: QueueType::Compute,
        }
    ])
        .expect("Initialization should work");
    // Now try to obtain an instance of all requested queues. Note that we cannot assert these do not refer to the same physical queue,
    // as they might not.
    {
        let _graphics = context.exec.get_queue::<Graphics>().expect("Graphics queue should exist");
    }

    {
        let _transfer = context.exec.get_queue::<Transfer>().expect("Transfer queue should exist");
    }

    {
        let _compute = context.exec.get_queue::<Compute>().expect("Compute queue should exist");
    }
}
