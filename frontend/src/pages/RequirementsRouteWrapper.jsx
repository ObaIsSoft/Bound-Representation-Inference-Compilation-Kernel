import React from 'react';
import RequirementsGatheringPage from './RequirementsGatheringPage';

export default function RequirementsRouteWrapper(props) {
    // This wrapper can be extended for context, auth, etc.
    return <RequirementsGatheringPage {...props} />;
}
